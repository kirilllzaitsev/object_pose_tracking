import os
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
from pose_tracking.metrics import calc_metrics
from pose_tracking.models.matcher import HungarianMatcher
from pose_tracking.models.set_criterion import SetCriterion
from pose_tracking.utils.artifact_utils import save_results
from pose_tracking.utils.common import cast_to_numpy
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    cam_to_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
    rot_mat_from_6d,
    rotate_pts_batch,
)
from pose_tracking.utils.misc import print_cls, reduce_dict, reduce_metric
from pose_tracking.utils.pose import (
    convert_pose_axis_angle_to_matrix,
    convert_pose_quaternion_to_matrix,
    convert_r_t_to_rt,
)
from pose_tracking.utils.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
)
from tqdm.auto import tqdm


class Trainer:

    def __init__(
        self,
        model,
        device,
        hidden_dim,
        rnn_type,
        criterion_trans=None,
        criterion_rot=None,
        criterion_pose=None,
        writer=None,
        do_debug=False,
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        do_predict_3d_rot=False,
        use_rnn=True,
        use_obs_belief=True,
        world_size=1,
        do_log_every_ts=False,
        do_log_every_seq=True,
        do_print_seq_stats=False,
        use_ddp=False,
        use_prev_pose_condition=False,
        do_predict_rel_pose=False,
        do_predict_kpts=False,
        use_prev_latent=False,
        logger=None,
        vis_epoch_freq=None,
        do_vis=False,
        exp_dir=None,
        model_name=None,
    ):
        assert criterion_pose is not None or (
            criterion_rot is not None and criterion_trans is not None
        ), "Either pose or rot & trans criteria must be provided"

        self.do_debug = do_debug
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_rnn = use_rnn
        self.use_obs_belief = use_obs_belief
        self.do_log_every_ts = do_log_every_ts
        self.do_log_every_seq = do_log_every_seq
        self.use_ddp = use_ddp
        self.use_prev_pose_condition = use_prev_pose_condition
        self.do_predict_rel_pose = do_predict_rel_pose
        self.use_prev_latent = use_prev_latent
        self.do_predict_kpts = do_predict_kpts
        self.do_vis = do_vis
        self.do_print_seq_stats = do_print_seq_stats

        self.world_size = world_size
        self.logger = default_logger if logger is None else logger
        self.vis_epoch_freq = vis_epoch_freq
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.model = model
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.criterion_trans = criterion_trans
        self.criterion_rot = criterion_rot
        self.criterion_pose = criterion_pose
        self.writer = writer

        self.use_pose_loss = criterion_pose is not None
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn
        self.vis_dir = f"{self.exp_dir}/vis"

        self.processed_data = defaultdict(list)
        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.train_epoch_count = 0

    def __repr__(self):
        return print_cls(self, excluded_attrs=["processed_data", "model"])

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_stats = self.batched_seq_forward(
                batched_seq=batched_seq,
                optimizer=optimizer,
                save_preds=save_preds,
                preds_dir=preds_dir,
                stage=stage,
                do_vis=do_vis,
            )

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                running_stats[k] += v
                if self.do_log and self.do_log_every_seq:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            if self.do_print_seq_stats:
                seq_pbar.set_postfix({k: v / (seq_pack_idx + 1) for k, v in running_stats.items()})

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            if self.use_ddp:
                v = reduce_metric(v, world_size=self.world_size)
            running_stats[k] = v / len(loader)

        if self.do_log:
            for k, v in running_stats.items():
                self.writer.add_scalar(f"{stage}_epoch/{k}", v, self.train_epoch_count)

        return running_stats

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
    ):

        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)
        pose_to_mat_converter_fn = (
            convert_pose_axis_angle_to_matrix if self.do_predict_3d_rot else convert_pose_quaternion_to_matrix
        )

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq))

        if do_opt_in_the_end:
            optimizer.zero_grad()
            total_loss = 0

        if self.use_ddp:
            self.model.module.reset_state(batch_size, device=self.device)
        else:
            self.model.reset_state(batch_size, device=self.device)
        if self.do_debug:
            self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 16)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        out_prev = None  # raw ouput of the model
        pose_mat_prev_gt_abs = None
        prev_latent = None
        nan_count = 0

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            h, w = rgb.shape[-2:]
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            if self.do_predict_rel_pose:
                if t == 0:
                    rot_prev_gt_abs = rot_gt_abs
                    if self.do_predict_3d_rot:
                        rot_prev_gt_abs = quaternion_to_axis_angle(rot_prev_gt_abs)
                    pose_prev_pred_abs = {"t": t_gt_abs, "rot": rot_prev_gt_abs}

                    pose_mat_prev_gt_abs = torch.stack([convert_pose_quaternion_to_matrix(rt) for rt in pose_gt_abs])

                    continue

            if self.use_prev_pose_condition:
                prev_pose = pose_prev_pred_abs if self.do_predict_rel_pose else out_prev
            else:
                prev_pose = None

            out = self.model(
                rgb,
                depth,
                bbox=bbox_2d,
                prev_pose=prev_pose,
                prev_latent=prev_latent,
            )

            # POSTPROCESS OUTPUTS

            rot_pred, t_pred = out["rot"], out["t"]

            if self.do_predict_2d_t:
                t_pred_2d_denorm = t_pred.detach().clone()
                t_pred_2d_denorm[:, 0] = t_pred_2d_denorm[:, 0] * w
                t_pred_2d_denorm[:, 1] = t_pred_2d_denorm[:, 1] * h

                center_depth_pred = out["center_depth"]
                t_pred_2d_backproj = []
                for sample_idx in range(batch_size):
                    t_pred_2d_backproj.append(
                        backproj_2d_to_3d(
                            t_pred_2d_denorm[sample_idx][None], center_depth_pred[sample_idx], intrinsics[sample_idx]
                        ).squeeze()
                    )
                t_pred = torch.stack(t_pred_2d_backproj).to(rot_pred.device)

            pose_mat_gt_abs = torch.stack([convert_pose_quaternion_to_matrix(rt) for rt in pose_gt_abs])
            rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]

            if self.do_predict_6d_rot:
                pose_mat_pred = torch.eye(4).repeat(batch_size, 1, 1).to(self.device)
                pose_mat_pred[:, :3, :3] = rot_mat_from_6d(rot_pred)
                pose_mat_pred[:, :3, 3] = t_pred
            else:
                pose_mat_pred = torch.stack(
                    [pose_to_mat_converter_fn(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
                )

            if self.do_predict_rel_pose:
                pose_mat_prev_pred_abs = torch.stack(
                    [
                        pose_to_mat_converter_fn(rt)
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
                t_gt_rel, rot_gt_rel_mat = pose_to_egocentric_delta_pose(pose_mat_gt_abs, pose_mat_prev_gt_abs)

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
                    t_gt_2d = cam_to_2d(t_gt_abs.unsqueeze(1), intrinsics).squeeze(1)
                    t_gt_2d_norm = t_gt_2d.clone()
                    t_gt_2d_norm[:, 0] = t_gt_2d_norm[:, 0] / w
                    t_gt_2d_norm[:, 1] = t_gt_2d_norm[:, 1] / h

                    t_pred_2d = out["t"]
                    depth_gt = t_gt_abs[:, 2]

                    loss_t_2d = self.criterion_trans(t_pred_2d, t_gt_2d_norm)
                    loss_center_depth = torch.abs(center_depth_pred - depth_gt).mean()
                    loss_t = loss_t_2d + loss_center_depth
                else:
                    if self.do_predict_rel_pose:
                        rel_t_scaler = 1
                        loss_t = self.criterion_trans(t_pred * rel_t_scaler, t_gt_rel * rel_t_scaler)
                    else:
                        loss_t = self.criterion_trans(t_pred_abs, t_gt_abs)

                # rot loss

                if self.do_predict_6d_rot:
                    loss_rot = torch.abs(
                        rotate_pts_batch(rot_mat_pred_abs, pts) - rotate_pts_batch(rot_mat_gt_abs, pts)
                    ).mean()
                else:
                    if self.do_predict_rel_pose:
                        if self.do_predict_rel_pose:
                            if self.do_predict_3d_rot:
                                rot_gt_rel = matrix_to_axis_angle(rot_gt_rel_mat)
                            else:
                                rot_gt_rel = matrix_to_quaternion(rot_gt_rel_mat)
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_rel)
                    else:
                        if self.do_predict_3d_rot:
                            rot_gt = quaternion_to_axis_angle(rot_gt_abs)
                        else:
                            rot_gt = rot_gt_abs
                        loss_rot = self.criterion_rot(rot_pred, rot_gt)
                loss = loss_rot + loss_t

            # depth loss
            if self.use_obs_belief:
                loss_depth = F.mse_loss(out["decoder_out"]["depth_final"], out["latent_depth"])
            else:
                loss_depth = torch.tensor(0.0).to(self.device)
            loss += loss_depth

            # priv loss
            if "priv_decoded" in out:
                loss_priv = compute_chamfer_dist(out["priv_decoded"], batch_t["priv"])
                loss += loss_priv * 0.01

            # kpt loss
            if self.do_predict_kpts:
                kpts_pred = out["kpts"]
                kpts_gt = batch_t["bbox_2d_kpts"].float()
                loss_kpts = F.huber_loss(kpts_pred, kpts_gt)
                bbox_2d_kpts_collinear_idxs = batch_t["bbox_2d_kpts_collinear_idxs"]
                loss_cr = kpt_cross_ratio_loss(kpts_pred, bbox_2d_kpts_collinear_idxs)
                loss += loss_kpts
                loss += loss_cr * 0.01

            # optim
            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                if self.do_debug:
                    grad_norms = [
                        cast_to_numpy(p.grad.norm()) for n, p in self.model.named_parameters() if p.grad is not None
                    ]
                    self.processed_data["grad_norm"].append(sum(grad_norms) / len(grad_norms))
                    self.processed_data["grad_norms"].append(grad_norms)
                optimizer.step()
            elif do_opt_in_the_end:
                total_loss += loss

            # METRICS

            bbox_3d = batch_t["mesh_bbox"]
            diameter = batch_t["mesh_diameter"]
            m_batch = defaultdict(list)

            for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_mat_pred_abs, pose_mat_gt_abs)):
                m_sample = calc_metrics(
                    pred_rt=pred_rt,
                    gt_rt=gt_rt,
                    pts=pts[sample_idx],
                    class_name=None,
                    use_miou=True,
                    bbox_3d=bbox_3d[sample_idx],
                    diameter=diameter[sample_idx],
                    is_meters=True,
                    log_fn=print if self.logger is None else self.logger.warning,
                )
                for k, v in m_sample.items():
                    m_batch[k].append(v)
                if any(np.isnan(v) for v in m_sample.values()):
                    nan_count += 1

            m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
            for k, v in m_batch_avg.items():
                seq_metrics[k] += v

            # UPDATE VARS

            if self.do_predict_rel_pose:
                if self.do_predict_3d_rot:
                    rot_prev_pred_abs = matrix_to_axis_angle(rot_mat_pred_abs)
                else:
                    rot_prev_pred_abs = matrix_to_quaternion(rot_mat_pred_abs)
                pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
            else:
                pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}
            if self.do_predict_2d_t:
                pose_prev_pred_abs["center_depth"] = center_depth_pred
            pose_prev_pred_abs = {k: v.detach() for k, v in pose_prev_pred_abs.items()}

            pose_mat_prev_gt_abs = pose_mat_gt_abs
            out_prev = {"t": out["t"], "rot": out["rot"]}
            prev_latent = out["prev_latent"].detach() if self.use_prev_latent else None

            # OTHER

            seq_stats["loss"] += loss
            seq_stats["loss_depth"] += loss_depth
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose
            else:
                seq_stats["loss_rot"] += loss_rot
                seq_stats["loss_t"] += loss_t
            if "priv_decoded" in out:
                seq_stats["loss_priv"] += loss_priv
            if self.do_predict_kpts:
                seq_stats["loss_kpts"] += loss_kpts
                seq_stats["loss_cr"] += loss_cr

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, pose_mat_pred_abs, preds_dir)

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["rgb", "depth", "intrinsics"]
                for k in ["mask", "mesh_bbox", "pts"]:
                    if k in batch_t and len(batch_t[k]) > 0:
                        vis_keys.append(k)
                for k in vis_keys:
                    vis_data[k].append([batch_t[k][i].cpu() for i in vis_batch_idxs])
                vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

            if self.do_debug:
                # add everything to processed_data
                self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})
                self.processed_data["rgb"].append(rgb)
                self.processed_data["mask"].append(mask)
                self.processed_data["pose_gt_abs"].append(pose_gt_abs)
                self.processed_data["pose_mat_gt_abs"].append(pose_mat_gt_abs)
                self.processed_data["depth"].append(depth)
                self.processed_data["rot_pred"].append(rot_pred)
                self.processed_data["t_pred"].append(t_pred)
                if self.do_predict_2d_t:
                    self.processed_data["t_gt_2d_norm"].append(t_gt_2d_norm)
                    self.processed_data["depth_gt"].append(depth_gt)
                    self.processed_data["center_depth_pred"].append(center_depth_pred)
                    self.processed_data["t_pred_2d_denorm"].append(t_pred_2d_denorm)
                    self.processed_data["t_pred_2d"].append(t_pred_2d)
                if "priv_decoded" in out:
                    self.processed_data["priv_decoded"].append(out["priv_decoded"])
                self.processed_data["pose_mat_pred_abs"].append(pose_mat_pred_abs)
                self.processed_data["pose_prev_pred_abs"].append(pose_prev_pred_abs)
                self.processed_data["pts"].append(pts)
                self.processed_data["bbox_3d"].append(bbox_3d)
                self.processed_data["diameter"].append(diameter)
                self.processed_data["loss"].append(loss)
                self.processed_data["m_batch"].append(m_batch)
                self.processed_data["loss_depth"].append(loss_depth)
                self.processed_data["out_prev"].append(out_prev)
                if self.use_pose_loss:
                    self.processed_data["loss_pose"].append(loss_pose)
                else:
                    self.processed_data["loss_rot"].append(loss_rot)
                    self.processed_data["loss_t"].append(loss_t)
                if self.do_predict_rel_pose:
                    if self.use_pose_loss:
                        self.processed_data["pose_mat_pred"].append(pose_mat_pred)
                        self.processed_data["pose_mat_gt_rel"].append(pose_mat_gt_rel)
                    else:
                        self.processed_data["t_gt_rel"].append(t_gt_rel)
                        self.processed_data["rot_gt_rel"].append(rot_gt_rel)

        if do_opt_in_the_end:
            total_loss /= seq_length
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = v / seq_length

        if nan_count > 0:
            seq_metrics["nan_count"] = nan_count

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}.pt"
            torch.save(vis_data, save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }


class TrainerDeformableDETR:

    def __init__(
        self,
        model,
        device,
        hidden_dim,
        rnn_type,
        num_classes,
        aux_loss,
        num_dec_layers,
        criterion_trans=None,
        criterion_rot=None,
        criterion_pose=None,
        writer=None,
        do_debug=False,
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        do_predict_3d_rot=False,
        use_rnn=True,
        use_obs_belief=True,
        world_size=1,
        do_log_every_ts=False,
        do_log_every_seq=True,
        do_print_seq_stats=False,
        use_ddp=False,
        use_prev_pose_condition=False,
        do_predict_rel_pose=False,
        do_predict_kpts=False,
        use_prev_latent=False,
        do_calibrate_kpt=False,
        logger=None,
        vis_epoch_freq=None,
        do_vis=False,
        exp_dir=None,
        model_name=None,
        focal_alpha=0.25,
        kpt_spatial_dim=2,
    ):
        assert criterion_pose is not None or (
            criterion_rot is not None and criterion_trans is not None
        ), "Either pose or rot & trans criteria must be provided"

        self.do_debug = do_debug
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_rnn = use_rnn
        self.use_obs_belief = use_obs_belief
        self.do_log_every_ts = do_log_every_ts
        self.do_log_every_seq = do_log_every_seq
        self.use_ddp = use_ddp
        self.use_prev_pose_condition = use_prev_pose_condition
        self.do_predict_rel_pose = do_predict_rel_pose
        self.use_prev_latent = use_prev_latent
        self.do_predict_kpts = do_predict_kpts
        self.do_vis = do_vis
        self.do_print_seq_stats = do_print_seq_stats

        self.world_size = world_size
        self.logger = default_logger if logger is None else logger
        self.vis_epoch_freq = vis_epoch_freq
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.model = model
        self.device = device
        self.kpt_spatial_dim = kpt_spatial_dim
        self.do_calibrate_kpt = do_calibrate_kpt
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.criterion_trans = criterion_trans
        self.criterion_rot = criterion_rot
        self.criterion_pose = criterion_pose
        self.writer = writer

        cost_class, cost_bbox, cost_giou = (2, 5, 2)
        self.matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.losses = ["labels", "boxes", "cardinality"]
        self.weight_dict = {
            "loss_ce": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_ce_enc": 1,
            "loss_bbox_enc": 5,
            "loss_giou_enc": 2,
        }
        if aux_loss:
            aux_weight_dict = {}
            for i in range(num_dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            aux_weight_dict.update({f"{k}_enc": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)
        self.criterion = SetCriterion(num_classes, self.matcher, self.weight_dict, self.losses, focal_alpha=focal_alpha)

        self.use_pose_loss = criterion_pose is not None
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn
        self.vis_dir = f"{self.exp_dir}/vis"

        self.processed_data = defaultdict(list)
        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.train_epoch_count = 0

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_stats = self.batched_seq_forward(
                batched_seq=batched_seq,
                optimizer=optimizer,
                save_preds=save_preds,
                preds_dir=preds_dir,
                stage=stage,
                do_vis=do_vis,
            )

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                running_stats[k] += v
                if self.do_log and self.do_log_every_seq:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            if self.do_print_seq_stats:
                seq_pbar.set_postfix({k: v / (seq_pack_idx + 1) for k, v in running_stats.items()})

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            if self.use_ddp:
                v = reduce_metric(v, world_size=self.world_size)
            running_stats[k] = v / len(loader)

        if self.do_log:
            for k, v in running_stats.items():
                self.writer.add_scalar(f"{stage}_epoch/{k}", v, self.train_epoch_count)

        return running_stats

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
    ):

        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)
        pose_to_mat_converter_fn = (
            convert_pose_axis_angle_to_matrix if self.do_predict_3d_rot else convert_pose_quaternion_to_matrix
        )

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq))

        if do_opt_in_the_end:
            optimizer.zero_grad()
            total_loss = 0

        # if self.use_ddp:
        #     self.model.module.reset_state(batch_size, device=self.device)
        # else:
        #     self.model.reset_state(batch_size, device=self.device)
        if self.do_debug:
            self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 16)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        out_prev = None  # raw ouput of the model
        pose_mat_prev_gt_abs = None
        prev_latent = None
        nan_count = 0

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            class_id = batch_t["class_id"]
            h, w = rgb.shape[-2:]
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            targets = {
                "labels": [v if v.ndim > 0 else v[None] for v in class_id],
                "boxes": [v.float() for v in [v if v.ndim > 1 else v[None] for v in bbox_2d]],
                "masks": mask,
            }
            if self.model_name == "detr_kpt":
                extra_kwargs = {}
                if self.do_calibrate_kpt:
                    extra_kwargs["intrinsics"] = intrinsics
                if self.kpt_spatial_dim > 2:
                    extra_kwargs["depth"] = depth
                out = self.model(rgb, **extra_kwargs)
            else:
                out = self.model(rgb)

            # POSTPROCESS OUTPUTS

            # LOSSES

            loss_dict = self.criterion(out, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            # optim
            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
            elif do_opt_in_the_end:
                total_loss += loss

            # METRICS

            bbox_3d = batch_t["mesh_bbox"]
            diameter = batch_t["mesh_diameter"]
            m_batch = defaultdict(list)

            m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
            for k, v in m_batch_avg.items():
                seq_metrics[k] += v

            # UPDATE VARS

            # OTHER

            seq_stats["loss"] += losses_reduced_scaled.item()
            for k, v in {**loss_dict_reduced_scaled, **loss_dict_reduced_unscaled}.items():
                seq_stats[k] += v
            seq_stats["class_error"] += loss_dict_reduced["class_error"]

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, pose_mat_pred_abs, preds_dir)

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["rgb", "depth", "intrinsics"]
                for k in ["mask", "mesh_bbox", "pts", "class_id"]:
                    if k in batch_t and len(batch_t[k]) > 0:
                        vis_keys.append(k)
                for k in vis_keys:
                    vis_data[k].append([batch_t[k][i].cpu() for i in vis_batch_idxs])
                # vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

            if self.do_debug:
                # add everything to processed_data
                self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})
                self.processed_data["rgb"].append(rgb)
                self.processed_data["mask"].append(mask)
                # self.processed_data["pose_gt_abs"].append(pose_gt_abs)
                # self.processed_data["pose_mat_gt_abs"].append(pose_mat_gt_abs)
                self.processed_data["depth"].append(depth)
                self.processed_data["rot_pred"].append(rot_pred)
                self.processed_data["t_pred"].append(t_pred)
                if self.do_predict_2d_t:
                    self.processed_data["t_gt_2d_norm"].append(t_gt_2d_norm)
                if "priv_decoded" in out:
                    self.processed_data["priv_decoded"].append(out["priv_decoded"])
                # self.processed_data["pose_mat_pred_abs"].append(pose_mat_pred_abs)
                # self.processed_data["pose_prev_pred_abs"].append(pose_prev_pred_abs)
                self.processed_data["pts"].append(pts)
                self.processed_data["bbox_3d"].append(bbox_3d)
                self.processed_data["diameter"].append(diameter)
                self.processed_data["loss"].append(loss)
                self.processed_data["m_batch"].append(m_batch)
                self.processed_data["loss_depth"].append(loss_depth)
                self.processed_data["out_prev"].append(out_prev)
                if self.use_pose_loss:
                    self.processed_data["loss_pose"].append(loss_pose)
                else:
                    self.processed_data["loss_rot"].append(loss_rot)
                    self.processed_data["loss_t"].append(loss_t)
                if self.do_predict_rel_pose:
                    self.processed_data["t_gt_rel"].append(t_gt_rel)
                    self.processed_data["rot_gt_rel"].append(rot_gt_rel)

        if do_opt_in_the_end:
            total_loss /= seq_length
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = v / seq_length

        if nan_count > 0:
            seq_metrics["nan_count"] = nan_count

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}.pt"
            torch.save(vis_data, save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }


class TrainerVideopose(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        from pizza.lib.losses.rotation import measure_rotation
        from pizza.lib.losses.translation import measure_translation

        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)

        # each batched seq is processed at once
        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts
        do_vis = False
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        MSELoss = torch.nn.MSELoss()

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_size = len(batched_seq["rgb"][0])
            for k, v in batched_seq.items():
                if k in ["rgb_path", "obj_name"]:
                    continue
                if isinstance(v, list):
                    if len(v) > 0:
                        if isinstance(v[0][0], torch.Tensor):
                            batched_seq[k] = torch.stack([torch.stack(vv) for vv in v]).to(self.device)
                        else:
                            if isinstance(v[0][0], np.ndarray):
                                batched_seq[k] = from_numpy(v).to(self.device)
                            else:
                                batched_seq[k] = torch.tensor(v).to(self.device)
            batched_seq = transfer_batch_to_device(batched_seq, self.device)
            res = self.model(batched_seq["rgb"])
            rot = res["rot"]
            delta_uv = res["delta_uv"]
            delta_depth = res["delta_depth"]

            batch = batched_seq
            ratio = batch["ratio"]
            # TODO: incorrect data prep. uv_first_frame should be of shape (B, 2) and not (B, L, 2)
            uv_first_frame = batch["uv_first_frame"]
            gt_delta_uv = batch["delta_uv"]
            gt_delta_depth = batch["delta_depth"]
            depth_first_frame = batch["depth_first_frame"]
            gt_delta_rotation = batch["gt_delta_rotation"]
            rotation_first_frame = batch["rotation_first_frame"]
            gt_rotations = batch["gt_rotations"]

            T_metrics = measure_translation(
                delta_uv_pred=delta_uv,
                delta_d_pred=delta_depth,
                uv_first_frame=uv_first_frame,
                d_first_frame=depth_first_frame,
                gt_delta_uv=gt_delta_uv,
                gt_delta_d=gt_delta_depth,
                alpha_resize=ratio,
                cumulative=True,
                loss_function=torch.nn.MSELoss(reduction="sum").cuda(depth_first_frame.get_device()),
                dataset_name="test",
            )
            R_metrics = measure_rotation(
                axis_angles_pred=rot,
                axis_angles_gt=gt_delta_rotation,
                rot_first_frame=rotation_first_frame,
                rots_gt=gt_rotations,
                cumulative=True,
            )

            loss_Z = MSELoss(delta_depth, gt_delta_depth) * (10**3)  # to mm
            loss_UV = MSELoss(delta_uv, gt_delta_uv) * (10**3)  # to mm
            loss = loss_Z + loss_UV + R_metrics["geodesic_err"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Call step of optimizer to update model params
                optimizer.step()

            losses = {
                "loss": loss,
                "loss_z": loss_Z,
                "loss_uv": loss_UV,
            }
            for k, v in {**T_metrics, **R_metrics, **losses}.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                    if self.use_ddp:
                        reduce_metric(v, world_size=self.world_size)
                running_stats[k] += v
                if self.do_log:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])

        return running_stats


class TrainerPizza(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
    ):
        from pizza.lib.losses.rotation import measure_rotation
        from pizza.lib.losses.translation import measure_translation

        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)

        # each batched seq is processed at once
        is_train = optimizer is not None
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts
        do_vis = False
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        MSELoss = torch.nn.MSELoss()

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_size = len(batched_seq["rgb"][0])
            for k, v in batched_seq.items():
                if k in ["rgb_path", "obj_name"]:
                    continue
                if isinstance(v, list):
                    if len(v) > 0:
                        if isinstance(v[0][0], torch.Tensor):
                            batched_seq[k] = torch.stack([torch.stack(vv) for vv in v]).to(self.device)
                        else:
                            if isinstance(v[0][0], np.ndarray):
                                batched_seq[k] = from_numpy(v).to(self.device)
                            else:
                                batched_seq[k] = torch.tensor(v).to(self.device)
            batched_seq = transfer_batch_to_device(batched_seq, self.device)
            res = self.model(batched_seq["rgb"])
            rot = res["rot"]
            delta_uv = res["delta_uv"]
            delta_depth = res["delta_depth"]

            batch = extend_seq_with_pizza_args(batched_seq)
            ratio = batch["ratio"]
            uv_first_frame = batch["uv_first_frame"]
            gt_delta_uv = batch["delta_uv"]
            gt_delta_depth = batch["delta_depth"]
            depth_first_frame = batch["depth_first_frame"]
            gt_delta_rotation = batch["gt_delta_rotation"]
            rotation_first_frame = batch["rotation_first_frame"]
            gt_rotations = batch["gt_rotations"]

            T_metrics = measure_translation(
                delta_uv_pred=delta_uv,
                delta_d_pred=delta_depth,
                uv_first_frame=uv_first_frame,
                d_first_frame=depth_first_frame,
                gt_delta_uv=gt_delta_uv,
                gt_delta_d=gt_delta_depth,
                alpha_resize=ratio,
                cumulative=True,
                loss_function=torch.nn.MSELoss(reduction="sum").cuda(depth_first_frame.get_device()),
                dataset_name="test",
                img_crop_size=2,
            )
            R_metrics = measure_rotation(
                axis_angles_pred=rot,
                axis_angles_gt=gt_delta_rotation,
                rot_first_frame=rotation_first_frame,
                rots_gt=gt_rotations,
                cumulative=True,
            )

            loss_z = MSELoss(delta_depth, gt_delta_depth) * (10**3)  # to mm
            loss_uv = MSELoss(delta_uv, gt_delta_uv) * (10**3)  # to mm
            loss = loss_z + loss_uv + R_metrics["geodesic_err"]

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Call step of optimizer to update model params
                optimizer.step()

            losses = {
                "loss": loss,
                "loss_z": loss_z,
                "loss_uv": loss_uv,
            }
            prefs = ["t_", "rot_", ""]
            for idx, metrics in enumerate([T_metrics, R_metrics, losses]):
                pref = prefs[idx]
                for k, v in metrics.items():
                    if "_err" not in k and not k.startswith("loss"):
                        continue
                    k = f"{pref}{k}"
                    if "cumulative" in k:
                        if k.startswith("rot_"):
                            k = "r_err"
                            v = v[-1]
                        if k.startswith("t_"):
                            k = "t_err"
                    if isinstance(v, torch.Tensor):

                        v = v.item()
                    if self.use_ddp:
                        reduce_metric(v, world_size=self.world_size)
                    running_stats[k] += v
                    if self.do_log:
                        self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                pose_mat_pred_abs = torch.stack([convert_pose_quaternion_to_matrix(rt) for rt in res["pose"]])
                save_results(batch, pose_mat_pred_abs, preds_dir)

        return running_stats
