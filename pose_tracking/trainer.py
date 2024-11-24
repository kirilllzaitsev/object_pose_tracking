import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from pose_tracking.config import default_logger
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.losses import compute_chamfer_dist, kpt_cross_ratio_loss
from pose_tracking.metrics import calc_metrics
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    cam_to_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
    rot_mat_from_6d,
    rotate_pts_batch,
)
from pose_tracking.utils.pipe_utils import reduce_metric, save_results
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
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
        use_ddp=False,
        use_prev_pose_condition=False,
        do_predict_rel_pose=False,
        do_predict_kpts=False,
        logger=None,
        vis_epoch_freq=None,
        do_vis=False,
        exp_dir=None,
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
        self.world_size = world_size
        self.logger = default_logger if logger is None else logger
        self.vis_epoch_freq = vis_epoch_freq
        self.do_log_every_ts = do_log_every_ts
        self.do_log_every_seq = do_log_every_seq
        self.use_ddp = use_ddp
        self.use_prev_pose_condition = use_prev_pose_condition
        self.do_predict_rel_pose = do_predict_rel_pose
        self.do_predict_kpts = do_predict_kpts
        self.do_vis = do_vis
        self.exp_dir = exp_dir

        self.use_pose_loss = criterion_pose is not None
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn

        self.model = model
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.criterion_trans = criterion_trans
        self.criterion_rot = criterion_rot
        self.criterion_pose = criterion_pose
        self.writer = writer
        self.processed_data = defaultdict(list)

        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.train_epoch_count = 0
        self.vis_dir = f"{self.exp_dir}/vis"

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
                if isinstance(v, torch.Tensor):
                    v = v.item()
                    if self.use_ddp:
                        reduce_metric(v, world_size=self.world_size)
                running_stats[k] += v
                if self.do_log:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            seq_pbar.set_postfix(
                {k: v / (seq_pack_idx + 1) for k, v in running_stats.items()},
            )

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            running_stats[k] = v / len(loader)

        if self.do_log and self.do_log_every_seq:
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

        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

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
            vis_batch_idxs = np.random.choice(batch_size, 2, replace=False)
            vis_data = defaultdict(list)

        abs_prev_pose = None  # the processed ouput of the model that matches model's expected format
        prev_model_out = None  # the raw ouput of the model
        prev_gt_mat = None  # prev gt in matrix form
        nan_count = 0

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["rgb"]
            seg_masks = batch_t["mask"]
            pose_gt = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            h, w = rgb.shape[-2:]
            t_gt = pose_gt[:, :3]
            rot_gt = pose_gt[:, 3:]

            if abs_prev_pose is None and self.do_predict_rel_pose:
                assert (
                    not self.do_predict_6d_rot and not self.do_predict_2d_t
                ), "Relative pose prediction is not supported with 6d rot or 2d t"
                abs_prev_pose = {"t": pose_gt[:, :3], "rot": pose_gt[:, 3:]}

            outputs = self.model(rgb, depth, prev_pose=abs_prev_pose if self.do_predict_rel_pose else prev_model_out)

            # POSTPROCESS OUTPUTS

            prev_model_out = {"t": outputs["t"], "rot": outputs["rot"]}
            rot_pred, t_pred = outputs["rot"], outputs["t"]

            if self.do_predict_6d_rot:
                rot_pred = rot_mat_from_6d(rot_pred)
            elif self.do_predict_3d_rot:
                rot_pred = axis_angle_to_matrix(rot_pred)

            if self.do_predict_2d_t:
                t_pred_2d_denorm = t_pred.detach().clone()
                t_pred_2d_denorm[:, 0] = t_pred_2d_denorm[:, 0] * w
                t_pred_2d_denorm[:, 1] = t_pred_2d_denorm[:, 1] * h

                depth_gt = t_gt[:, 2]
                center_depth_pred = outputs["center_depth"]
                t_pred_2d_backproj = []
                for sample_idx in range(len(depth_gt)):
                    t_pred_2d_backproj.append(
                        backproj_2d_to_3d(
                            t_pred_2d_denorm[sample_idx][None], center_depth_pred[sample_idx], intrinsics[sample_idx]
                        ).squeeze()
                    )
                t_pred = torch.stack(t_pred_2d_backproj).to(rot_pred.device)

            pose_gt_mat = torch.stack([convert_pose_quaternion_to_matrix(rt) for rt in pose_gt])
            prev_gt_mat = pose_gt_mat
            if self.do_predict_6d_rot or self.do_predict_3d_rot:
                pose_pred = torch.eye(4).repeat(batch_size, 1, 1).to(self.device)
                pose_pred[:, :3, :3] = rot_pred
                pose_pred[:, :3, 3] = t_pred
            else:
                pose_pred = torch.stack(
                    [convert_pose_quaternion_to_matrix(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
                )

            if self.do_predict_rel_pose:
                prev_pose_mat = torch.stack(
                    [
                        convert_pose_quaternion_to_matrix(rt)
                        for rt in torch.cat([abs_prev_pose["t"], abs_prev_pose["rot"]], dim=1)
                    ]
                )
                pose_pred = egocentric_delta_pose_to_pose(
                    prev_pose_mat,
                    trans_delta=pose_pred[:, :3, 3],
                    rot_mat_delta=pose_pred[:, :3, :3],
                    do_couple_rot_t=False,
                )

            if self.do_predict_rel_pose:
                abs_rot_quat = matrix_to_quaternion(pose_pred[:, :3, :3])
                abs_prev_pose = {"t": pose_pred[:, :3, 3], "rot": abs_rot_quat}
            else:
                abs_prev_pose = {"t": t_pred, "rot": rot_pred}
                if self.do_predict_2d_t:
                    abs_prev_pose["center_depth"] = center_depth_pred
            abs_prev_pose = {k: v.detach() for k, v in abs_prev_pose.items()}

            # LOSSES
            # -- pose_pred is abs, t_pred/rot_pred can be rel or abs

            if self.use_pose_loss:
                loss_pose = self.criterion_pose(pose_pred, pose_gt_mat, pts)
                loss = loss_pose.clone()
            else:
                if self.do_predict_rel_pose:
                    t_gt_rel, rot_gt_rel = pose_to_egocentric_delta_pose(pose_gt_mat, prev_gt_mat)
                    rot_gt_rel = matrix_to_quaternion(rot_gt_rel)
                if self.do_predict_2d_t:
                    t_gt_2d = cam_to_2d(t_gt.unsqueeze(1), intrinsics).squeeze(1)
                    t_gt_2d_norm = t_gt_2d.clone()
                    t_gt_2d_norm[:, 0] = t_gt_2d_norm[:, 0] / w
                    t_gt_2d_norm[:, 1] = t_gt_2d_norm[:, 1] / h

                    t_pred_2d = outputs["t"]
                    loss_t_2d = torch.abs(t_pred_2d - t_gt_2d_norm).mean()
                    loss_center_depth = torch.abs(center_depth_pred - depth_gt).mean()

                    loss_t = loss_t_2d + loss_center_depth
                else:
                    if self.do_predict_rel_pose:
                        loss_t = self.criterion_trans(t_pred, t_gt_rel)
                    else:
                        loss_t = self.criterion_trans(t_pred, t_gt)
                if self.do_predict_6d_rot:
                    loss_rot = torch.abs(
                        rotate_pts_batch(pose_pred[:, :3, :3], pts) - rotate_pts_batch(pose_gt_mat[:, :3, :3], pts)
                    ).mean()
                elif self.do_predict_3d_rot:
                    # rot_gt_3d = quaternion_to_axis_angle(rot_gt)
                    loss_rot = F.mse_loss(rot_pred, pose_gt_mat[:, :3, :3])
                else:
                    if self.do_predict_rel_pose:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_rel)
                    else:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt)
                loss = loss_rot + loss_t

            if self.use_obs_belief:
                loss_depth = F.mse_loss(outputs["decoder_out"]["depth_final"], outputs["latent_depth"])
            else:
                loss_depth = torch.tensor(0.0).to(self.device)
            loss += loss_depth

            if "priv_decoded" in outputs:
                loss_priv = compute_chamfer_dist(outputs["priv_decoded"], batch_t["priv"])
                loss += loss_priv * 0.01

            if self.do_predict_kpts:
                kpts_pred = outputs["kpts"]
                kpts_gt = batch_t["bbox_2d_kpts"].float()
                loss_kpts = F.huber_loss(kpts_pred, kpts_gt)
                bbox_2d_kpts_collinear_idxs = batch_t["bbox_2d_kpts_collinear_idxs"]
                loss_cr = kpt_cross_ratio_loss(kpts_pred, bbox_2d_kpts_collinear_idxs)
                loss += loss_kpts
                loss += loss_cr * 0.01

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
            for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_pred, pose_gt_mat)):
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

            # OTHER

            seq_stats["loss"] += loss
            seq_stats["loss_depth"] += loss_depth
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose
            else:
                seq_stats["loss_rot"] += loss_rot
                seq_stats["loss_t"] += loss_t
            if "priv_decoded" in outputs:
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
                save_results(batch_t, t_pred, rot_pred, preds_dir)

            if do_vis:
                # save inputs to the exp dir
                for k in ["rgb", "mask", "depth", "intrinsics", "mesh_bbox"]:
                    vis_data[k].append([batch_t[k][i] for i in vis_batch_idxs])
                vis_data["pose_pred"].append(pose_pred[vis_batch_idxs].detach())
                vis_data["pose_gt_mat"].append(pose_gt_mat[vis_batch_idxs])

            if self.do_debug:
                # add everything to processed_data
                self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})
                self.processed_data["rgb"].append(rgb)
                self.processed_data["seg_masks"].append(seg_masks)
                self.processed_data["pose_gt"].append(pose_gt)
                self.processed_data["pose_gt_mat"].append(pose_gt_mat)
                self.processed_data["depth"].append(depth)
                self.processed_data["rot_pred"].append(rot_pred)
                self.processed_data["t_pred"].append(t_pred)
                if self.do_predict_2d_t:
                    self.processed_data["t_gt_2d_norm"].append(t_gt_2d_norm)
                if "priv_decoded" in outputs:
                    self.processed_data["priv_decoded"].append(outputs["priv_decoded"])
                self.processed_data["pose_pred"].append(pose_pred)
                self.processed_data["abs_prev_pose"].append(abs_prev_pose)
                self.processed_data["pts"].append(pts)
                self.processed_data["bbox_3d"].append(bbox_3d)
                self.processed_data["diameter"].append(diameter)
                self.processed_data["loss"].append(loss)
                self.processed_data["m_batch"].append(m_batch)
                self.processed_data["loss_depth"].append(loss_depth)
                self.processed_data["prev_model_out"].append(prev_model_out)
                if self.use_pose_loss:
                    self.processed_data["loss_pose"].append(loss_pose)
                else:
                    self.processed_data["loss_rot"].append(loss_rot)
                    self.processed_data["loss_t"].append(loss_t)

        if do_opt_in_the_end:
            total_loss /= len(batched_seq)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                stats[k] = v / len(batched_seq)

        if nan_count > 0:
            seq_metrics["nan_count"] = nan_count

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = f"{self.vis_dir}/epoch_{self.train_epoch_count}.pt"
            torch.save(vis_data, save_vis_path)
            self.logger.info(f"Saved vis data to {save_vis_path}")

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
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        return running_stats
