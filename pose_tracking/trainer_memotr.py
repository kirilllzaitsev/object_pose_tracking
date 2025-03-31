import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.trainer_detr import TrainerDeformableDETR
from pose_tracking.utils.artifact_utils import save_results_v2
from pose_tracking.utils.common import detach_and_cpu, extract_idxs
from pose_tracking.utils.detr_utils import postprocess_detr_outputs
from pose_tracking.utils.geom import convert_2d_t_to_3d, egocentric_delta_pose_to_pose
from pose_tracking.utils.misc import is_empty, reduce_dict
from pose_tracking.utils.pose import convert_r_t_to_rt
from pose_tracking.utils.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
)
from tqdm import tqdm

try:
    from memotr.structures.track_instances import TrackInstances
    from memotr.submit_engine import filter_by_area, filter_by_score
    from memotr.utils.nested_tensor import tensor_list_to_nested_tensor
except:
    pass


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

    def get_param_dicts(self):
        from memotr.train_engine import get_param_groups

        param_groups, lr_names = get_param_groups(config=self.config, model=self.model)
        return param_groups

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

        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq["image"][0])
        batch_size = len(batched_seq["image"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

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

        for t in ts_pbar:
            batch_t = {}
            for k, v in batched_seq.items():
                if not is_empty(v):
                    if isinstance(v, list):
                        batch_t[k] = [x[t] for x in v]
                    else:
                        batch_t[k] = v[:, t]

            rgb = batch_t["image"]
            targets = batch_t["target"]
            # pose_gt_abs = torch.stack([x["pose"] for x in targets])
            intrinsics = [x["intrinsics"] for x in targets]
            pts = batch_t["mesh_pts"]
            h, w = rgb.shape[-2:]

            model_forward_res = self.model_forward(
                batch_t,
                tracks=tracks,
            )
            out = model_forward_res["out"]

            criterion_res = self.criterion.process_single_frame(
                model_outputs=out, tracked_instances=tracks, frame_idx=t
            )
            previous_tracks, new_tracks, unmatched_dets, indices = (
                criterion_res["tracked_instances"],
                criterion_res["new_trackinstances"],
                criterion_res["unmatched_detections"],
                criterion_res["matched_idxs"],
            )
            if is_train:
                if t < seq_length - 1:
                    tracks = self.model_without_ddp.postprocess_single_frame(
                        previous_tracks, new_tracks, unmatched_dets
                    )
            else:
                # raise RuntimeError
                tracks: list[TrackInstances] = self.model_without_ddp.postprocess_single_frame(
                    previous_tracks, new_tracks, None
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

            loss_dict = model_forward_res["loss_dict"]
            loss = self.criterion.get_sum_loss_dict(loss_dict=loss_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced_scaled = reduce_dict(loss_dict)
            loss_dict_reduced_scaled["loss_bbox"] = loss_dict_reduced_scaled.pop("box_l1_loss")
            loss_dict_reduced_scaled["loss_giou"] = loss_dict_reduced_scaled.pop("box_giou_loss")
            loss_dict_reduced_scaled["loss_ce"] = loss_dict_reduced_scaled.pop("label_focal_loss")
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
                pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in target_rts])

                if len(target_rts) == 0:
                    ...
                else:

                    t_pred = torch.cat([track.ts[matched_idx_pred[tidx]] for tidx, track in enumerate(tracks)], dim=0)
                    rot_pred = torch.cat(
                        [track.rots[matched_idx_pred[tidx]] for tidx, track in enumerate(tracks)], dim=0
                    )

                    if self.do_predict_2d_t:
                        center_depth_pred = out["center_depth"][idx]
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred, center_depth_pred, intrinsics, hw=(h, w)
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

                    batch_t = self.prepare_batch_t_for_metrics_mot(batch_t)
                    m_batch_avg = self.calc_metrics_batch(
                        batch_t, pose_mat_pred_metrics=pose_mat_pred_metrics, pose_mat_gt_metrics=pose_mat_gt_metrics
                    )
                    for k, v in m_batch_avg.items():
                        if "classes" in k:
                            continue
                        seq_metrics[k].append(v)

            # UPDATE VARS

            if self.use_pose and seq_length > 1:
                if self.do_predict_rel_pose:
                    if self.do_predict_3d_rot:
                        rot_prev_pred_abs = matrix_to_axis_angle(rot_mat_pred_abs)
                    elif self.do_predict_6d_rot:
                        rot_prev_pred_abs = matrix_to_rotation_6d(rot_mat_pred_abs)
                    else:
                        rot_prev_pred_abs = matrix_to_quaternion(rot_mat_pred_abs)
                    pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
                else:
                    pose_prev_pred_abs = {}
                #     pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}
                if self.do_predict_2d_t:
                    pose_prev_pred_abs["center_depth"] = center_depth_pred
                pose_prev_pred_abs = {k: v.detach() for k, v in pose_prev_pred_abs.items()}

                pose_mat_prev_gt_abs = pose_mat_gt_abs

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
                target_sizes = torch.stack([x["size"] for x in batch_t["target"]])
                out_formatted = postprocess_detr_outputs(
                    out, target_sizes=target_sizes, is_focal_loss=self.args.tf_use_focal_loss
                )
                bboxs = []
                labels = []
                for bidx, out_b in enumerate(out_formatted):
                    keep = out_b["scores"].cpu() > out_b["scores_no_object"].cpu()
                    # keep = torch.ones_like(res['scores']).bool()
                    if sum(keep) == 0:
                        print(f"{bidx=} failed")
                        continue
                    boxes_b = out_b["boxes"][keep]
                    labels_b = out_b["labels"][keep]
                    bboxs.append(boxes_b)
                    labels.append(labels_b)
                save_results_v2(
                    rgb,
                    intrinsics=intrinsics,
                    pose_gt=pose_mat_gt_abs,
                    pose_pred=pose_mat_pred_abs,
                    rgb_path=batch_t["rgb_path"],
                    preds_dir=preds_dir,
                    bboxs=bboxs,
                    labels=labels,
                )

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["image", "mesh_bbox", "mask", "depth"]
                for k in vis_keys:
                    if len(batch_t.get(k, [])) == 0:
                        continue
                    vis_data[k].append([batch_t[k][i].cpu() for i in vis_batch_idxs])
                vis_data["targets"].append(extract_idxs(targets, vis_batch_idxs))
                vis_data["out"].append(
                    extract_idxs(
                        detach_and_cpu({k: v for k, v in out.items() if k not in ["det_query_embed"]}),
                        vis_batch_idxs,
                        do_extract_dict_contents=True,
                    )
                )
                if self.model_name == "detr_kpt":
                    vis_data["kpts"].append(extract_idxs(out["kpts"], vis_batch_idxs))
                    vis_data["descriptors"].append(extract_idxs(out["descriptors"], vis_batch_idxs))
                if self.use_pose:
                    vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs[vis_batch_idxs]))

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
            metrics_abs = self.calc_metrics_batch(batch_t, pose_mat_pred_abs, pose_mat_gt_abs)
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

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }

    def model_forward(self, batch_t, tracks, prev_features=None, **kwargs):
        frame = batch_t["image"]
        for f in frame:
            f.requires_grad_(False)
        # check frame grads
        frame = tensor_list_to_nested_tensor(tensor_list=frame).to(self.device)
        out = self.model(frame=frame, tracks=tracks)
        loss_dict, _ = self.criterion.get_mean_by_n_gts()
        return {
            "out": out,
            "loss_dict": loss_dict,
        }
