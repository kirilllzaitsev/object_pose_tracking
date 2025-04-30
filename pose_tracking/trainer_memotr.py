import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.models.encoders import is_param_part_of_encoders
from pose_tracking.trainer_detr import TrainerDeformableDETR
from pose_tracking.utils.artifact_utils import save_results_v2
from pose_tracking.utils.common import cast_to_numpy, detach_and_cpu, extract_idxs
from pose_tracking.utils.detr_utils import (
    postprocess_detr_boxes,
    postprocess_detr_outputs,
)
from pose_tracking.utils.geom import (
    allocentric_to_egocentric,
    convert_2d_t_to_3d,
    egocentric_delta_pose_to_pose,
)
from pose_tracking.utils.misc import is_empty, reduce_dict
from pose_tracking.utils.pose import convert_r_t_to_rt
from pose_tracking.utils.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
)
from tqdm import tqdm

from memotr.models.runtime_tracker import RuntimeTracker

try:
    from memotr.structures.track_instances import TrackInstances
    from memotr.submit_engine import filter_by_area, filter_by_score
    from memotr.utils.nested_tensor import tensor_list_to_nested_tensor
except:
    pass


def filter_by_score(tracks: TrackInstances, thresh: float = 0.7):
    keep = torch.max(tracks.scores, dim=-1).values > thresh
    return tracks[keep]


class TrainerMemotr(TrainerDeformableDETR):

    def __init__(
        self,
        *args,
        config,
        **kwargs,
    ):

        self.config = config
        super().__init__(*args, **kwargs)

        self.result_score_thresh = 0.5
        self.result_area_thresh = 100
        self.det_score_thresh = 0.5
        self.track_score_thresh = 0.5

        TrackInstances.rot_out_dim = self.config["rot_out_dim"]

    def get_param_dicts(self):
        from memotr.train_engine import get_param_groups

        param_groups, lr_names = get_param_groups(config=self.config, model=self.model)
        return param_groups

    def freeze_encoder(self, model_without_ddp):
        for name, p in model_without_ddp.named_parameters():
            if (
                is_param_part_of_encoders(name, self.encoder_module_prefix)
                or ("transformer" in name and "bbox_embed" not in name)
                or any(x in name for x in ["feature_projs", "decoder", "class_embed"])
            ):
                p.requires_grad = False

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
        last_step_state=None,
    ):

        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq["image"][0])
        batch_size = len(batched_seq["image"])

        seq_stats = defaultdict(list)
        seq_metrics = defaultdict(list)
        ts_pbar = tqdm(range(seq_length), desc="Timestep", leave=False, total=len(batched_seq), disable=True)

        total_losses = []

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        pose_mat_prev_gt_abs = None
        pose_tokens_per_layer = None
        prev_tokens = None
        nan_count = 0
        do_skip_first_step = False

        tracks = TrackInstances.init_tracks(
            batch=batched_seq,
            hidden_dim=self.model_without_ddp.hidden_dim,
            num_classes=self.model_without_ddp.num_classes,
            device=self.device,
            use_dab=self.config["USE_DAB"],
            rot_out_dim=self.config["rot_out_dim"],
            t_out_dim=self.config["t_out_dim"],
            WITH_BOX_REFINE=self.config["WITH_BOX_REFINE"],
        )
        self.criterion.log = {}
        self.criterion.init_a_clip(
            batch=batched_seq,
            hidden_dim=self.model_without_ddp.hidden_dim,
            num_classes=self.model_without_ddp.num_classes,
            device=self.device,
        )
        batched_seq = transfer_batch_to_device(batched_seq, self.device)
        failed_ts = []

        is_test = stage == "test"
        if is_test:
            tracker = RuntimeTracker(
                det_score_thresh=0.5,
                track_score_thresh=0.5,
                miss_tolerance=30,
                use_motion=False,
                motion_min_length=0,
                motion_max_length=0,
                visualize=False,
                use_dab=self.config["USE_DAB"],
                matcher=self.criterion.matcher,
            )

        for t in ts_pbar:
            batch_t = self.get_seq_slice_for_dict_seq(batched_seq, t)

            # print(f"{batch_t['rgb_path']}")
            rgb = batch_t["image"]
            targets = batch_t["target"]
            # pose_gt_abs = torch.stack([x["pose"] for x in targets])
            intrinsics = [x["intrinsics"] for x in targets]
            h, w = rgb.shape[-2:]

            model_forward_res = self.model_forward(
                batch_t,
                tracks=tracks,
            )
            out = model_forward_res["out"]

            if not is_test:
                try:
                    criterion_res = self.criterion.process_single_frame(
                        model_outputs=out, tracked_instances=tracks, frame_idx=t
                    )
                except Exception as e:
                    print(f"Error in process_single_frame: {e}")
                    print(f"{out=}")
                    print(f"{tracks=}")
                    print(f"{t=}")
                    print(f"{batch_t=}")
                    raise e
                previous_tracks, new_tracks, unmatched_dets, indices = (
                    criterion_res["tracked_instances"],
                    criterion_res["new_trackinstances"],
                    criterion_res["unmatched_detections"],
                    criterion_res["matched_idxs"],
                )
                # TODO: why -1 in original? they don't calc metrics, hence do not need tracks
                if t < seq_length:
                    tracks = self.model_without_ddp.postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets
                    )
            else:
                previous_tracks2, new_tracks2 = tracker.update(model_outputs=out, tracks=tracks, batch=batch_t)
                tracks: list[TrackInstances] = self.model_without_ddp.postprocess_single_frame(
                    previous_tracks2, new_tracks2, None
                )
                # tracks_result = tracks[0].to(torch.device("cpu"))
                # tracks_result = filter_by_score(tracks_result, thresh=self.result_score_thresh)
                # TODO: proper eval?
                # ori_h, ori_w = height, width
                # tracks_result.area = (
                #     tracks_result.boxes[:, 2] * ori_w * tracks_result.boxes[:, 3] * ori_h
                # )
                # tracks_result = filter_by_area(tracks_result, thresh=result_area_thresh)

            # POSTPROCESS OUTPUTS

            ...

            # LOSSES

            loss_dict, _ = self.criterion.get_mean_by_n_gts()
            loss = self.criterion.get_sum_loss_dict(loss_dict=loss_dict)

            # reduce dict can be costly
            loss_dict_reduced_scaled = loss_dict
            loss_dict_reduced_scaled["loss_bbox"] = loss_dict_reduced_scaled.pop("box_l1_loss")
            loss_dict_reduced_scaled["loss_giou"] = loss_dict_reduced_scaled.pop("box_giou_loss")
            loss_dict_reduced_scaled["loss_ce"] = loss_dict_reduced_scaled.pop("label_focal_loss")
            loss_dict_reduced_scaled["loss_rot"] = loss_dict_reduced_scaled.pop("rot_loss")
            loss_dict_reduced_scaled["loss_t"] = loss_dict_reduced_scaled.pop("t_loss")
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            # optim
            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if self.do_debug or do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    if do_vis:
                        vis_data["grad_norm"].append(grad_norm)
                        vis_data["grad_norms"].append(grad_norms)

                optimizer.step()
                optimizer.zero_grad()
            elif do_opt_in_the_end:
                total_losses.append(loss)

            # METRICS

            if self.use_pose:
                matched_idx_tgt = [track.matched_idx[track.matched_idx >= 0] for track in tracks]
                matched_idx_pred = [torch.nonzero(track.matched_idx >= 0, as_tuple=True)[0] for track in tracks]
                target_rts = torch.cat(
                    [torch.cat([t["t"][i], t["rot"][i]], dim=1) for t, i in zip(targets, matched_idx_tgt)], dim=0
                )

                if len(target_rts) == 0:
                    pose_mat_gt_abs = torch.empty(0, 4, 4).to(self.device)
                    pose_mat_pred_abs = torch.empty(0, 4, 4).to(self.device)
                    if self.do_predict_2d_t:
                        center_depth_pred = torch.empty(0, 1).to(self.device)
                else:
                    pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in target_rts])
                    t_pred = torch.cat([track.ts[matched_idx_pred[tidx]] for tidx, track in enumerate(tracks)], dim=0)
                    rot_pred = torch.cat(
                        [track.rots[matched_idx_pred[tidx]] for tidx, track in enumerate(tracks)], dim=0
                    )
                    other_values_for_metrics = self.get_req_target_values_for_metrics(targets, matched_idx_tgt)

                    if self.do_predict_2d_t:
                        center_depth_pred = t_pred[..., 2:]
                        t_pred_2d = t_pred[..., :2]
                        intrinsics_rep = []
                        for bidx in range(batch_size):
                            intrinsics_rep.extend(
                                intrinsics[bidx].unsqueeze(0).repeat(len(matched_idx_pred[bidx]), 1, 1)
                            )
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred_2d, center_depth_pred, intrinsics_rep, hw=(h, w)
                        )
                        t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

                    pred_rts = torch.cat([t_pred, rot_pred], dim=1)
                    pose_mat_pred = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pred_rts])

                    if self.do_predict_rel_pose:
                        pose_mat_prev_pred_abs = torch.stack(
                            [
                                self.pose_to_mat_converter_fn(rt)
                                for rt in torch.cat([pose_prev_pred_abs["t"], pose_prev_pred_abs["rot"]], dim=1)
                            ]
                        )
                        pose_mat_pred_abs = egocentric_delta_pose_to_pose(
                            pose_mat_prev_pred_abs,
                            trans_delta=pose_mat_pred[:, :3, 3],
                            rot_mat_delta=pose_mat_pred[:, :3, :3],
                            do_couple_rot_t=False,
                        )
                    else:
                        pose_mat_pred_abs = pose_mat_pred
                    t_pred_abs = pose_mat_pred_abs[:, :3, 3]
                    rot_mat_pred_abs = pose_mat_pred_abs[:, :3, :3]

                    # since only pair seq for now
                    if self.do_predict_rel_pose:
                        pose_mat_pred_metrics = pose_mat_pred
                        # TODO: multi-object case
                        pose_mat_gt_rel = torch.stack(
                            [self.pose_to_mat_converter_fn(target["pose_rel"][0]) for target in targets]
                        )
                        rot_mat_gt_rel = pose_mat_gt_rel[:, :3, :3]
                        t_gt_rel = pose_mat_gt_rel[:, :3, 3]
                        pose_mat_gt_metrics = convert_r_t_to_rt(rot_mat_gt_rel, t_gt_rel)
                    else:
                        pose_mat_pred_metrics = pose_mat_pred_abs
                        pose_mat_gt_metrics = pose_mat_gt_abs

                    # TODO: matched_idx_tgt should idx within batch as well as the batch itself
                    m_batch_avg = self.calc_metrics_batch_mot(
                        pose_mat_pred_metrics=pose_mat_pred_metrics,
                        pose_mat_gt_metrics=pose_mat_gt_metrics,
                        **other_values_for_metrics,
                    )
                    for k, v in m_batch_avg.items():
                        if "classes" in k:
                            continue
                        seq_metrics[k].append(v)

            # temporal loss
            if self.use_temporal_loss and t > 0:
                # match poses from prev and new tracks
                prev_track_poses = []
                new_track_poses = []
                for bidx, new_track in enumerate(tracks):
                    prev_track = prev_tracks[bidx]
                    matched_tids = [tid for tid in new_track.ids if tid != -1 and tid in prev_track.ids]
                    if len(matched_tids) > 0:
                        new_matched_id_idxs = [i for i, tid in enumerate(new_track.ids) if tid in matched_tids]
                        prev_matched_id_idxs = [i for i, tid in enumerate(prev_track.ids) if tid in matched_tids]
                        new_track_poses.extend(
                            self.parse_track_pose(new_track, intrinsics, hw=(h, w))[new_matched_id_idxs]
                        )
                        prev_track_poses.extend(
                            self.parse_track_pose(prev_track, intrinsics, hw=(h, w))[prev_matched_id_idxs]
                        )

                if len(prev_track_poses) > 0:
                    prev_track_poses = torch.stack(prev_track_poses)
                    new_track_poses = torch.stack(new_track_poses)
                    delta_pose = pose_to_egocentric_delta_pose_mat(prev_track_poses, new_track_poses)
                    delta_pose_lie = Se3.from_matrix(delta_pose).log()
                    loss_temporal = 0.1 * F.mse_loss(delta_pose_lie, torch.zeros_like(delta_pose_lie))
                    loss += loss_temporal
                    seq_stats["loss_temporal"].append(loss_temporal.item())

            # UPDATE VARS

            prev_tracks = [t.clone() for t in tracks]

            # OTHER

            loss_value = losses_reduced_scaled.item()
            seq_stats["loss"].append(loss_value)
            for k, v in {**loss_dict_reduced_scaled}.items():
                if "indices" in k:
                    continue
                seq_stats[k].append(v.item())
            # for k in ["class_error", "cardinality_error"]:
            #     if k in loss_dict_reduced:
            #         v = loss_dict_reduced[k]
            #         seq_stats[k].append(v.item())

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", np.mean(v), self.ts_counts_per_stage[stage])

            if not math.isfinite(loss_value):
                self.logger.error(f"Loss is {loss_value}, stopping training")
                # self.logger.error(loss_dict_reduced)
                if t > 0:
                    self.logger.error(f"{batched_seq[t-1]=}")
                self.logger.error(f"{batched_seq[t]=}")
                if self.use_pose:
                    self.logger.error(f"rot_pred: {rot_pred}")
                self.logger.error(f"seq_metrics: {seq_metrics}")
                self.logger.error(f"seq_stats: {seq_stats}")
                sys.exit(1)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert self.use_pose
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                target_sizes = torch.stack([x["size"] for x in batch_t["target"]]).cpu()
                # out_formatted = postprocess_detr_outputs(
                #     out, target_sizes=target_sizes, is_focal_loss=self.args.tf_use_focal_loss
                # )
                # bboxs = []
                # labels = []
                # scores = []
                # for bidx, out_b in enumerate(out_formatted):
                #     keep = out_b["scores"].cpu() > out_b["scores_no_object"].cpu()
                #     # keep = torch.ones_like(res['scores']).bool()
                #     if sum(keep) == 0:
                #         failed_ts.append(t)
                #         continue
                #     boxes_b = out_b["boxes"][keep]
                #     scores_b = out_b["scores"][keep]
                #     labels_b = out_b["labels"][keep]
                #     bboxs.append(boxes_b)
                #     labels.append(labels_b)
                #     scores.append(scores_b)
                tracks_result = tracks[0].to(torch.device("cpu"))
                if len(tracks_result.ids) > 1:
                    print(f"{t=} {len(tracks_result.ids)=} {tracks_result.ids=}")
                # ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
                # box = [x, y, w, h]
                # tracks_result.area = tracks_result.boxes[:, 2] * ori_w * \
                #                      tracks_result.boxes[:, 3] * ori_h
                # tracks_result = filter_by_area(tracks_result)
                result_score_thresh = self.result_score_thresh
                tracks_result = filter_by_score(tracks_result, thresh=result_score_thresh)
                # to xyxy:
                tracks_result.boxes = postprocess_detr_boxes(tracks_result.boxes, target_sizes=target_sizes)
                det_res = {
                    "bbox": tracks_result.boxes,
                    "labels": tracks_result.labels,
                    "scores": torch.max(tracks_result.scores, dim=-1).values,
                    "track_ids": tracks_result.ids,
                }
                track_ts = tracks_result.ts
                if len(track_ts) > 0:
                    if self.do_predict_2d_t:
                        center_depth_pred = track_ts[..., 2:]
                        t_pred_2d = track_ts[..., :2]
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred_2d, center_depth_pred, [i.cpu() for i in intrinsics], hw=(h, w)
                        )
                        track_ts = convert_2d_t_pred_to_3d_res["t_pred"]
                else:
                    failed_ts.append(t)

                track_rots = tracks_result.rots
                pose_mat_pred_abs = self.pose_to_mat_converter_fn(torch.cat([track_ts, track_rots], dim=-1))
                if self.model_without_ddp.use_roi:
                    pose_mat_pred_abs = allocentric_to_egocentric(cast_to_numpy(pose_mat_pred_abs))
                save_results_v2(
                    rgb,
                    intrinsics=intrinsics,
                    pose_gt=pose_mat_gt_abs,
                    pose_pred=pose_mat_pred_abs,
                    rgb_path=batch_t["rgb_path"],
                    preds_dir=preds_dir,
                    det_res=det_res,
                    mesh_bbox=tracks_result.other_attrs["mesh_bbox"],
                )

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["image", "mesh_bbox", "mask", "depth"]
                for k in vis_keys:
                    if len(batch_t.get(k, [])) == 0:
                        continue
                    vis_data[k].append(detach_and_cpu(batch_t[k]))
                vis_data["targets"].append(detach_and_cpu(targets))
                vis_data["out"].append(
                    detach_and_cpu({k: v for k, v in out.items() if k not in ["det_query_embed"]}),
                )
                if self.use_pose and len(pose_mat_pred_abs) > 0:
                    vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs))
                    vis_data["pose_mat_gt_abs"].append(detach_and_cpu(pose_mat_gt_abs))

                # vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

        if do_opt_in_the_end:
            total_loss = torch.mean(torch.stack(total_losses))
            total_loss.backward()
            unused_params = []
            if len(unused_params):
                self.logger.error(f"Unused params: {unused_params}")
                sys.exit(1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if self.do_debug or do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                if do_vis:
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
            optimizer.step()
            optimizer.zero_grad()

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = np.mean(v)

        if self.use_pose and self.do_predict_rel_pose:
            # calc loss/metrics btw accumulated abs poses
            metrics_abs = self.calc_metrics_batch_mot(pose_mat_pred_abs, pose_mat_gt_abs, **other_values_for_metrics)
            for k, v in metrics_abs.items():
                seq_metrics[f"{k}_abs"].append(v)
            if not self.include_abs_pose_loss_for_rel:
                with torch.no_grad():

                    t_gt_abs = pose_mat_gt_abs[:, :3, 3]
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_gt_abs = self.rot_mat_to_vector_converter_fn(rot_mat_gt_abs)
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}.pt"
            torch.save(vis_data, save_vis_path)
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        if len(failed_ts) > 0:
            self.logger.error(f"Failed steps: {failed_ts}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }

    def get_seq_slice_for_dict_seq(self, batched_seq, t):
        batch_t = {}
        for k, v in batched_seq.items():
            if not is_empty(v):
                if isinstance(v, list):
                    batch_t[k] = [x[t] for x in v]
                else:
                    batch_t[k] = v[:, t]
        return batch_t

    def model_forward(self, batch_t, tracks, prev_features=None, **kwargs):
        frame = batch_t["image"]
        for f in frame:
            f.requires_grad_(False)
        # check frame grads
        frame = tensor_list_to_nested_tensor(tensor_list=frame).to(self.device)
        out = self.model(frame=frame, tracks=tracks)
        return {
            "out": out,
        }
