import functools
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pose_tracking.config import default_logger
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.dataset.ds_common import from_numpy
from pose_tracking.dataset.pizza_utils import extend_seq_with_pizza_args
from pose_tracking.losses import compute_chamfer_dist, kpt_cross_ratio_loss
from pose_tracking.metrics import (
    calc_metrics,
    calc_r_error,
    calc_t_error,
    eval_batch_det,
)
from pose_tracking.models.encoders import FrozenBatchNorm2d
from pose_tracking.models.matcher import HungarianMatcher
from pose_tracking.models.set_criterion import SetCriterion
from pose_tracking.trainer import Trainer
from pose_tracking.utils.artifact_utils import save_results, save_results_v2
from pose_tracking.utils.common import cast_to_numpy, detach_and_cpu, extract_idxs
from pose_tracking.utils.detr_utils import postprocess_detr_outputs
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    cam_to_2d,
    convert_2d_t_to_3d,
    convert_3d_t_for_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
    rot_mat_from_6d,
    rotate_pts_batch,
)
from pose_tracking.utils.kpt_utils import (
    get_pose_from_3d_2d_matches,
    get_pose_from_matches,
    kabsch_torch_batched,
)
from pose_tracking.utils.misc import (
    match_module_by_name,
    print_cls,
    reduce_dict,
    reduce_metric,
    split_arr,
)
from pose_tracking.utils.pose import convert_pose_vector_to_matrix, convert_r_t_to_rt
from pose_tracking.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class TrainerKeypoints(Trainer):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

        is_train = stage == "train"
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(list)
        seq_metrics = defaultdict(list)
        ts_pbar = tqdm(
            enumerate(batched_seq),
            desc="Timestep",
            leave=False,
            total=len(batched_seq),
            disable=True,
        )

        total_losses = []

        if self.do_debug and "lstm" in self.model_name:
            self.processed_data["state"].append(detach_and_cpu({"hx": self.model.hx, "cx": self.model.cx}))

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None
        out_prev = None  # raw ouput of the model at prev step
        pose_mat_prev_gt_abs = None
        prev_latent = None
        prev_pose = None
        state = None
        do_skip_first_step = False
        prev_kpts = None
        kpts_key = "bbox_3d_kpts" if self.bbox_num_kpts == 32 else "bbox_3d_kpts_corners"

        for t, batch_t in ts_pbar:
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            features_rgb = batch_t.get("features_rgb")
            h, w = rgb.shape[-2:]
            hw = (h, w)
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            if self.do_predict_rel_pose:
                if t == 0 and last_step_state is not None:
                    pose_mat_prev_gt_abs = last_step_state["pose_mat_prev_gt_abs"]
                    pose_prev_pred_abs = last_step_state["pose_prev_pred_abs"]
                    prev_latent = last_step_state.get("prev_latent")
                    state = last_step_state.get("state")
                elif t == 0:
                    pose_mat_prev_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])

                    out = self.model(
                        rgb=rgb,
                        depth=depth,
                        bbox=bbox_2d,
                        state=None,
                        features_rgb=features_rgb,
                        bbox_kpts=batch_t[kpts_key],
                    )
                    state = out["state"]

                    prev_latent = out["latent"] if self.use_prev_latent else None
                    prev_kpts = out["kpts"]

                    pose_prev_pred_abs = {"t": t_gt_abs, "rot": rot_gt_abs}

                    if save_preds:
                        assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                        save_results(batch_t, pose_mat_prev_gt_abs, preds_dir, gt_pose=pose_mat_prev_gt_abs)

                    continue

            if self.use_prev_pose_condition and self.do_predict_rel_pose:
                prev_pose = pose_prev_pred_abs

            out = self.model(
                rgb=rgb,
                depth=depth,
                bbox=bbox_2d,
                prev_pose=prev_pose,
                prev_latent=prev_latent,
                state=state,
                features_rgb=features_rgb,
                bbox_kpts=batch_t[kpts_key],
                prev_bbox_kpts=batched_seq[t - 1][kpts_key],
            )

            # POSTPROCESS OUTPUTS

            rot_pred, t_pred = out["rot"], out["t"]
            state = out["state"]
            kpts_pred = out["kpts"]

            if self.do_predict_2d_t:
                center_depth_pred = out["center_depth"]
                convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(t_pred, center_depth_pred, intrinsics.float(), hw=hw)
                t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

            pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])
            rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]

            # pose_mat_pred = torch.stack(
            #     [self.pose_to_mat_converter_fn(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
            # )
            if self.do_predict_rel_pose:
                bbox_3d_kpts_prev = batched_seq[t - 1][kpts_key]
                bbox_3d_kpts_cur = bbox_3d_kpts_prev + kpts_pred
                ka_rot, ka_t, _ = kabsch_torch_batched(bbox_3d_kpts_prev, bbox_3d_kpts_cur)
            else:
                ka_rot_bidxs = []
                ka_t_bidxs = []
                for bidx in range(batch_size):
                    res = get_pose_from_3d_2d_matches(
                        kpts_3d=batch_t["bbox_3d_kpts_mesh"][bidx],
                        kpts_2d=cam_to_2d(kpts_pred, intrinsics.float())[bidx],
                        intrinsics=intrinsics[bidx].float(),
                    )
                    # cur_pose_mat_from_rel = convert_r_t_to_rt(res["R"], res["t"].squeeze())
                    ka_t_bidxs.append(torch.tensor(res["t"].squeeze(), device=self.device))
                    ka_rot_bidxs.append(torch.tensor(res["R"], device=self.device))
                ka_rot = torch.stack(ka_rot_bidxs)
                ka_t = torch.stack(ka_t_bidxs)

            rot_pred = self.rot_mat_to_vector_converter_fn(ka_rot)
            t_pred = ka_t
            out["t"] = t_pred
            out["rot"] = rot_pred
            pose_mat_pred = torch.eye(4, device=self.device)[None].repeat(batch_size, 1, 1)
            pose_mat_pred[:, :3, :3] = ka_rot
            pose_mat_pred[:, :3, 3] = ka_t

            rot_mat_pred = pose_mat_pred[:, :3, :3]

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
            if self.do_predict_rel_pose:
                t_gt_rel, rot_gt_rel_mat = pose_to_egocentric_delta_pose(pose_mat_prev_gt_abs, pose_mat_gt_abs)

            # LOSSES
            # -- t_pred/rot_pred can be rel or abs

            if self.use_pose_loss:
                if self.do_predict_rel_pose:
                    pose_mat_gt_rel = convert_r_t_to_rt(rot_gt_rel_mat, t_gt_rel)
                    loss_pose = self.criterion_pose(pose_mat_pred, pose_mat_gt_rel, pts)
                else:
                    loss_pose = self.criterion_pose(pose_mat_pred_abs, pose_mat_gt_abs, pts)
                loss = loss_pose.clone()
            else:
                # t loss

                if self.do_predict_2d_t:
                    t_pred_2d = out["t"]
                    if self.do_predict_rel_pose:
                        loss_uv = self.criterion_trans(t_pred_2d[:, :2], t_gt_rel[:, :2])
                        # trickier for depth (should be change in scale)
                        loss_z = self.criterion_trans(center_depth_pred, t_gt_rel[:, 2:3])
                    else:
                        t_gt_2d_norm, depth_gt = convert_3d_t_for_2d(t_gt_abs, intrinsics, hw)
                        loss_uv = self.criterion_trans(t_pred_2d, t_gt_2d_norm)
                        loss_z = self.criterion_trans(center_depth_pred, depth_gt)
                    loss_t = loss_uv + loss_z
                else:
                    if self.do_predict_rel_pose:
                        rel_t_scaler = 1
                        loss_t = self.criterion_trans(t_pred * rel_t_scaler, t_gt_rel * rel_t_scaler)
                    else:
                        loss_t = self.criterion_trans(t_pred_abs, t_gt_abs)

                # rot loss

                if self.criterion_rot_name == "displacement":
                    self.criterion_rot = functools.partial(self.criterion_rot, pts=pts)

                if self.do_predict_rel_pose:
                    if self.use_rot_mat_for_loss:
                        loss_rot = self.criterion_rot(rot_mat_pred, rot_gt_rel_mat)
                    else:
                        rot_gt_rel = self.rot_mat_to_vector_converter_fn(rot_gt_rel_mat)
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_rel)
                else:
                    if self.use_rot_mat_for_loss:
                        loss_rot = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_abs)

                if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    loss_t += loss_t_abs / seq_length
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    loss_rot += loss_rot_abs / seq_length

                if self.opt_only is None:
                    loss = self.tf_t_loss_coef * loss_t + self.tf_rot_loss_coef * loss_rot
                else:
                    loss = 0
                    # assert any(x in self.opt_only for x in ["rot", "t"]), f"Invalid opt_only: {self.opt_only}"
                    if "rot" in self.opt_only:
                        loss += loss_rot * self.tf_rot_loss_coef
                    if "t" in self.opt_only:
                        loss += loss_t * self.tf_t_loss_coef

            # depth loss
            if self.use_belief_decoder:
                loss_depth = F.mse_loss(out["decoder_out"]["depth_final"], out["latent_depth"])
                loss += loss_depth
            else:
                loss_depth = torch.tensor(0.0).to(self.device)

            # priv loss
            if "priv_decoded" in out:
                loss_priv = compute_chamfer_dist(out["priv_decoded"], batch_t["priv"])
                loss += loss_priv * 0.01

            # kpt loss
            kpts_gt = batch_t[kpts_key].float()
            if self.do_predict_rel_pose:
                kpts_gt_prev = batched_seq[t - 1][kpts_key].float()
                kpts_gt_delta = kpts_gt - kpts_gt_prev
                loss_kpts = F.huber_loss(kpts_pred, kpts_gt_delta)
                # bbox_2d_kpts_collinear_idxs = batch_t["bbox_2d_kpts_collinear_idxs"]
                # loss_cr = kpt_cross_ratio_loss(kpts_gt_prev + kpts_pred, bbox_2d_kpts_collinear_idxs)
                # loss += loss_cr
            else:
                loss_kpts = F.huber_loss(kpts_pred, kpts_gt)

            loss += loss_kpts

            if self.do_predict_rel_pose and self.do_predict_abs_pose:
                t_pred_abs_pose, rot_pred_abs_pose = out["t_abs_pose"], out["rot_abs_pose"]
                losses_abs_pose = self.calc_abs_pose_loss_for_rel(
                    pose_gt_abs, t_pred_abs_pose=t_pred_abs_pose, rot_pred_abs_pose=rot_pred_abs_pose
                )
                loss += losses_abs_pose["loss_abs_pose"]
                for k, v in losses_abs_pose.items():
                    seq_stats[k].append(v.item())

            # optim
            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    self.logger.warning("\n///")
                    self.logger.warning(f"\nSTEP: {self.ts_counts_per_stage[stage]=}\n{grad_norm=}\n///")
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                    self.logger.warning("///\n")
                optimizer.step()
                optimizer.zero_grad()
                state = [x if x is None else x.detach() for x in state]

            # METRICS

            if self.do_predict_rel_pose:
                pose_mat_pred_metrics = pose_mat_pred
                pose_mat_gt_metrics = convert_r_t_to_rt(rot_gt_rel_mat, t_gt_rel)
            else:
                pose_mat_pred_metrics = pose_mat_pred_abs
                pose_mat_gt_metrics = pose_mat_gt_abs

            m_batch_avg = self.calc_metrics_batch(
                batch_t, pose_mat_pred_metrics=pose_mat_pred_metrics, pose_mat_gt_metrics=pose_mat_gt_metrics
            )
            for k, v in m_batch_avg.items():
                seq_metrics[k].append(v)

            # OTHER

            seq_stats["loss"].append(loss.item())
            seq_stats["loss_depth"].append(loss_depth.item())
            if self.use_pose_loss:
                seq_stats["loss_pose"].append(loss_pose.item())
            else:
                seq_stats["loss_rot"].append(loss_rot.item())
                seq_stats["loss_t"].append(loss_t.item())
                if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())
            if "priv_decoded" in out:
                seq_stats["loss_priv"].append(loss_priv.item())
            seq_stats["loss_kpts"].append(loss_kpts.item())
            if self.do_predict_kpts:
                seq_stats["loss_cr"].append(loss_cr.item())
            if self.do_predict_2d_t:
                seq_stats["loss_uv"].append(loss_uv.item())
                seq_stats["loss_z"].append(loss_z.item())

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])
            if self.do_debug:
                for k, v in {**m_batch_avg, **{"loss_rot": loss_rot.item(), "loss_t": loss_t.item()}}.items():
                    self.seq_stats_all_seq[self.seq_counts_per_stage[stage]][k].append(v)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, pose_mat_pred_abs, preds_dir, gt_pose=pose_mat_gt_abs)

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["rgb", "intrinsics"]
                for k in ["mask", "mesh_bbox", "pts", "depth"]:
                    if k in batch_t and len(batch_t[k]) > 0:
                        vis_keys.append(k)
                for k in vis_keys:
                    vis_data[k].append([detach_and_cpu(batch_t[k][i]) for i in vis_batch_idxs])
                vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs[vis_batch_idxs]))
                vis_data["pose_mat_pred"].append(detach_and_cpu(pose_mat_pred[vis_batch_idxs]))
                vis_data["intrinsics"].append(detach_and_cpu(intrinsics[vis_batch_idxs]))
                vis_data["out"].append(
                    ({k: detach_and_cpu(v[vis_batch_idxs]) for k, v in out.items() if k in ["t", "rot"]})
                )
                vis_data["t_pred"].append(detach_and_cpu(t_pred[vis_batch_idxs]))
                vis_data["rot_pred"].append(detach_and_cpu(rot_pred[vis_batch_idxs]))
                vis_data["pose_mat_gt_abs"].append(detach_and_cpu(pose_mat_gt_abs[vis_batch_idxs]))

                if self.do_predict_2d_t:
                    if not self.do_predict_rel_pose:
                        vis_data["t_gt_2d_norm"].append(detach_and_cpu(t_gt_2d_norm))
                        vis_data["depth_gt"].append(detach_and_cpu(depth_gt))
                        vis_data["t_pred_2d_denorm"].append(
                            detach_and_cpu(convert_2d_t_pred_to_3d_res["t_pred_2d_denorm"])
                        )
                    vis_data["center_depth_pred"].append(detach_and_cpu(center_depth_pred))
                    vis_data["t_pred_2d"].append(detach_and_cpu(t_pred_2d))
                if "priv_decoded" in out:
                    vis_data["priv_decoded"].append(detach_and_cpu(out["priv_decoded"]))
                vis_data["pose_prev_pred_abs"].append(detach_and_cpu(pose_prev_pred_abs))
                vis_data["pts"].append(detach_and_cpu(pts))
                vis_data["bbox_3d"].append(detach_and_cpu(batch_t["mesh_bbox"]))
                vis_data["out_prev"].append(detach_and_cpu(out_prev))
                if self.do_predict_rel_pose:
                    if self.use_pose_loss:
                        vis_data["pose_mat_pred"].append(detach_and_cpu(pose_mat_pred))
                        vis_data["pose_mat_gt_rel"].append(detach_and_cpu(pose_mat_gt_rel))
                    else:
                        vis_data["t_gt_rel"].append(detach_and_cpu(t_gt_rel))
                        if not self.use_rot_mat_for_loss:
                            vis_data["rot_gt_rel"].append(detach_and_cpu(rot_gt_rel))
                        vis_data["rot_gt_rel_mat"].append(detach_and_cpu(rot_gt_rel_mat))
                        vis_data["pose_mat_prev_gt_abs"].append(detach_and_cpu(pose_mat_prev_gt_abs))
                if self.do_predict_kpts:
                    vis_data["kpts_pred"].append(detach_and_cpu(out["kpts"]))
                    vis_data["kpts_gt"].append(detach_and_cpu(batch_t["bbox_2d_kpts"]))
                    if self.use_pnp_for_rot_pred:
                        vis_data["prev_kpts"].append(detach_and_cpu(prev_kpts))

                if torch.isnan(loss):
                    if t > 0:
                        self.logger.error(f"{batched_seq[t-1]=}")
                    self.logger.error(f"{batched_seq[t]=}")
                    self.logger.error(f"{loss_t=}")
                    self.logger.error(f"{loss_rot=}")
                    self.logger.error(f"rot_pred: {rot_pred}")
                    self.logger.error(f"rot_mat_pred: {rot_mat_pred}")
                    self.logger.error(f"rot_gt_abs: {rot_gt_abs}")
                    self.logger.error(f"rot_mat_gt_abs: {rot_mat_gt_abs}")
                    self.logger.error(f"seq_metrics: {seq_metrics}")
                    self.logger.error(f"seq_stats: {seq_stats}")
                    if self.do_predict_rel_pose:
                        self.logger.error(f"rot_gt_rel: {rot_gt_rel_mat}")
                    sys.exit(1)

            # UPDATE VARS

            if self.do_predict_rel_pose:
                if self.do_predict_abs_pose:
                    pose_prev_pred_abs = {"t": t_pred_abs_pose, "rot": rot_pred_abs_pose}
                else:
                    rot_prev_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                    pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
            else:
                pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}

            pose_mat_prev_gt_abs = pose_mat_gt_abs
            out_prev = {"t": out["t"], "rot": out["rot"]}
            prev_latent = out["latent"] if self.use_prev_latent else None

        if self.use_rnn:
            last_step_state = {
                "prev_latent": prev_latent.detach() if self.use_prev_latent else None,
                "pose_prev_pred_abs": {k: v.detach() for k, v in pose_prev_pred_abs.items()},
                "pose_mat_prev_gt_abs": pose_mat_prev_gt_abs,
                "state": [x if x is None else x.detach() for x in state],
            }
        else:
            last_step_state = None

        if not self.do_debug:
            for stats in [seq_stats, seq_metrics]:
                for k, v in stats.items():
                    stats[k] = np.mean(v)

        if self.do_predict_rel_pose:
            # calc loss/metrics btw accumulated abs poses
            metrics_abs = self.calc_metrics_batch(batch_t, pose_mat_pred_abs, pose_mat_gt_abs)
            for k, v in metrics_abs.items():
                seq_metrics[f"{k}_abs"].append(v)
            if not self.include_abs_pose_loss_for_rel:
                with torch.no_grad():
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = (
                f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}_step_{self.ts_counts_per_stage[stage]}.pt"
            )
            vis_data["seq_metrics"].append(seq_metrics)
            vis_data["seq_stats"].append(seq_stats)
            torch.save(vis_data, save_vis_path)
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
            "last_step_state": last_step_state,
        }
