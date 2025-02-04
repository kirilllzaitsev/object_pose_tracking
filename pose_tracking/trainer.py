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


class Trainer:

    def __init__(
        self,
        model,
        device,
        hidden_dim,
        rnn_type,
        seq_len,
        criterion_trans=None,
        criterion_rot=None,
        criterion_rot_name=None,
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
        include_abs_pose_loss_for_rel=False,
        do_predict_rel_pose=False,
        do_predict_kpts=False,
        do_chunkify_val=False,
        use_prev_latent=False,
        do_perturb_init_gt_for_rel_pose=False,
        logger=None,
        vis_epoch_freq=None,
        do_vis=False,
        exp_dir=None,
        model_name=None,
        opt_only=None,
        max_clip_grad_norm=0.1,
        tf_t_loss_coef=1,
        tf_rot_loss_coef=1,
        use_entire_seq_in_train=False,
        **kwargs,
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
        self.do_chunkify_val = do_chunkify_val
        self.do_perturb_init_gt_for_rel_pose = do_perturb_init_gt_for_rel_pose
        self.use_entire_seq_in_train = use_entire_seq_in_train

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
        self.criterion_rot_name = criterion_rot_name
        self.criterion_pose = criterion_pose
        self.writer = writer
        self.seq_len = seq_len
        self.opt_only = opt_only
        self.max_clip_grad_norm = max_clip_grad_norm
        self.tf_t_loss_coef = tf_t_loss_coef
        self.tf_rot_loss_coef = tf_rot_loss_coef

        if self.use_ddp:
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        self.do_reset_state = "lstm" in model_name
        self.use_pose_loss = criterion_pose is not None
        self.include_abs_pose_loss_for_rel = include_abs_pose_loss_for_rel and do_predict_rel_pose
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn
        self.vis_dir = f"{self.exp_dir}/vis"
        self.use_rot_mat_for_loss = self.criterion_rot_name in ["displacement", "geodesic_mat"]
        self.save_vis_paths = []
        self.seq_stats_all_seq = defaultdict(lambda: defaultdict(list))

        self.processed_data = defaultdict(list)
        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.train_epoch_count = 0

        if self.do_predict_3d_rot:
            self.pose_to_mat_converter_fn = functools.partial(convert_pose_vector_to_matrix, rot_repr="axis_angle")
            self.rot_mat_to_vector_converter_fn = matrix_to_axis_angle
            self.rot_vector_to_mat_converter_fn = axis_angle_to_matrix
        elif self.do_predict_6d_rot:
            self.pose_to_mat_converter_fn = functools.partial(convert_pose_vector_to_matrix, rot_repr="rotation6d")
            self.rot_mat_to_vector_converter_fn = matrix_to_rotation_6d
            self.rot_vector_to_mat_converter_fn = rotation_6d_to_matrix
        else:
            self.pose_to_mat_converter_fn = convert_pose_vector_to_matrix
            self.rot_mat_to_vector_converter_fn = matrix_to_quaternion
            self.rot_vector_to_mat_converter_fn = quaternion_to_matrix

        if do_predict_3d_rot:
            assert criterion_rot_name not in ["geodesic", "geodesic_mat", "videopose"], criterion_rot_name

    def __repr__(self):
        return print_cls(self, excluded_attrs=["processed_data", "model", "args", "model_without_ddp"])

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
        seq_pbar = tqdm(loader, desc="Seq", leave=False, disable=len(loader) == 1)
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0 and stage == "train"

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):

            batch_size = len(batched_seq[0]["rgb"])

            if self.do_reset_state:
                if self.model_without_ddp.rnn_state_init_type in ["learned"]:
                    self.do_reset_state = False
                self.model_without_ddp.reset_state(batch_size, device=self.device)

            if (stage != "train" and not self.do_chunkify_val) or (stage == "train" and not self.use_entire_seq_in_train):
                seq_stats = self.batched_seq_forward(
                    batched_seq=batched_seq,
                    optimizer=optimizer,
                    save_preds=save_preds,
                    preds_dir=preds_dir,
                    stage=stage,
                    do_vis=do_vis,
                )
            else:
                batched_seq_chunks = split_arr(batched_seq, len(batched_seq) // self.seq_len)
                seq_stats = defaultdict(lambda: defaultdict(float))
                num_chunks = len(batched_seq_chunks)

                for cidx, chunk in tqdm(enumerate(batched_seq_chunks), desc="Subseq", leave=False, total=num_chunks):
                    seq_stats_chunk = self.batched_seq_forward(
                        batched_seq=chunk,
                        optimizer=optimizer,
                        save_preds=save_preds,
                        preds_dir=preds_dir,
                        stage=stage,
                        do_vis=do_vis,
                    )
                    for k, v in seq_stats_chunk.items():
                        for kk, vv in v.items():
                            seq_stats[k][kk] += vv
                for k, v in seq_stats.items():
                    for kk, vv in v.items():
                        seq_stats[k][kk] /= len(batched_seq_chunks)

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                running_stats[k] += v
                if self.do_log and self.do_log_every_seq:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            if self.do_print_seq_stats:
                seq_pbar.set_postfix({k: v / (seq_pack_idx + 1) for k, v in running_stats.items()})

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            running_stats[k] = v / len(loader)
        running_stats = reduce_dict(running_stats, device=self.device)

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

        is_train = stage == "train"
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(
            enumerate(batched_seq),
            desc="Timestep",
            leave=False,
            total=len(batched_seq),
            disable=True,
        )

        total_loss = 0

        if self.do_debug:
            self.processed_data["state"].append(detach_and_cpu({"hx": self.model.hx, "cx": self.model.cx}))

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        out_prev = None  # raw ouput of the model
        pose_mat_prev_gt_abs = None
        prev_latent = None
        state = None
        nan_count = 0

        for t, batch_t in ts_pbar:
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            h, w = rgb.shape[-2:]
            hw = (h, w)
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            if self.do_predict_rel_pose:
                if t == 0:
                    if self.do_perturb_init_gt_for_rel_pose:
                        noise_t = (
                            torch.rand_like(t_gt_abs)
                            * 0.02
                            * (torch.randint(0, 2, t_gt_abs.shape) * 2 - 1).to(self.device)
                        )
                        noise_rot_mat = torch.stack([torch.eye(3) for _ in range(batch_size)])
                        for i in range(batch_size):
                            j = torch.randint(0, 3, (1,))
                            angle = np.random.uniform(1) * 5
                            if j == 0:
                                angles = [angle, 0, 0]
                            elif j == 1:
                                angles = [0, angle, 0]
                            else:
                                angles = [0, 0, angle]

                            noise_rot_mat[i] = torch.tensor(
                                Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                            )
                        noise_rot_mat = noise_rot_mat.to(self.device)
                        rot_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])[
                            ..., :3, :3
                        ]
                        rot_mat_gt_abs = torch.bmm(rot_mat_gt_abs, noise_rot_mat)
                        rot_gt_abs = self.rot_mat_to_vector_converter_fn(rot_mat_gt_abs)
                    else:
                        noise_t = 0

                    pose_prev_pred_abs = {"t": t_gt_abs + noise_t, "rot": rot_gt_abs}

                    pose_mat_prev_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])

                    prev_latent = torch.cat(
                        [self.model_without_ddp.encoder_img(rgb), self.model_without_ddp.encoder_depth(depth)], dim=1
                    )

                    if save_preds:
                        assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                        save_results(batch_t, pose_mat_prev_gt_abs, preds_dir, gt_pose=pose_mat_prev_gt_abs)

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
                state=state,
            )

            # POSTPROCESS OUTPUTS

            rot_pred, t_pred = out["rot"], out["t"]
            state = out["state"]

            if self.do_predict_2d_t:
                center_depth_pred = out["center_depth"]
                convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(t_pred, center_depth_pred, intrinsics, hw=hw)
                t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

            pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])
            rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]

            pose_mat_pred = torch.stack(
                [self.pose_to_mat_converter_fn(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
            )
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
                        loss_z = self.criterion_trans(center_depth_pred, t_gt_rel[:, 2])
                        loss_t = loss_uv + loss_z
                    else:

                        t_gt_2d_norm, depth_gt = convert_3d_t_for_2d(t_gt_abs, intrinsics, hw)

                        loss_t_2d = self.criterion_trans(t_pred_2d, t_gt_2d_norm)
                        loss_center_depth = self.criterion_trans(center_depth_pred, depth_gt)
                        loss_t = loss_t_2d + loss_center_depth
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
                    loss_t += loss_t_abs
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    loss_rot += loss_rot_abs

                if self.opt_only is None:
                    loss = self.tf_t_loss_coef * loss_t + self.tf_rot_loss_coef * loss_rot
                else:
                    loss = 0
                    assert any(x in self.opt_only for x in ["rot", "t"]), f"Invalid opt_only: {self.opt_only}"
                    if "rot" in self.opt_only:
                        loss += loss_rot * self.tf_rot_loss_coef
                    if "t" in self.opt_only:
                        loss += loss_t * self.tf_t_loss_coef

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                optimizer.step()
                optimizer.zero_grad()
                state = [x if x is None else x.detach() for x in state]
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

            m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
            for k, v in m_batch_avg.items():
                seq_metrics[k] += v

            # OTHER

            seq_stats["loss"] += loss.item()
            seq_stats["loss_depth"] += loss_depth.item()
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose.item()
            else:
                seq_stats["loss_rot"] += loss_rot.item()
                seq_stats["loss_t"] += loss_t.item()
                if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                    seq_stats["loss_t_abs"] += loss_t_abs.item()
                    seq_stats["loss_rot_abs"] += loss_rot_abs.item()
            if "priv_decoded" in out:
                seq_stats["loss_priv"] += loss_priv.item()
            if self.do_predict_kpts:
                seq_stats["loss_kpts"] += loss_kpts.item()
                seq_stats["loss_cr"] += loss_cr.item()

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
                vis_keys = ["rgb", "depth", "intrinsics"]
                for k in ["mask", "mesh_bbox", "pts"]:
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
                vis_data["bbox_3d"].append(detach_and_cpu(bbox_3d))
                vis_data["m_batch"].append(detach_and_cpu(m_batch))
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
                rot_prev_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
            else:
                pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}
            if self.do_predict_2d_t:
                pose_prev_pred_abs["center_depth"] = center_depth_pred
            pose_prev_pred_abs = {k: v.detach() for k, v in pose_prev_pred_abs.items()}

            pose_mat_prev_gt_abs = pose_mat_gt_abs
            out_prev = {"t": out["t"], "rot": out["rot"]}
            prev_latent = out["prev_latent"].detach() if self.use_prev_latent else None

        num_steps = seq_length
        if self.do_predict_rel_pose:
            num_steps -= 1

        if do_opt_in_the_end:
            total_loss /= num_steps
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                vis_data["grad_norm"].append(grad_norm)
                vis_data["grad_norms"].append(grad_norms)
            optimizer.step()
            optimizer.zero_grad()

        self.model_without_ddp.set_state(state)
        self.model_without_ddp.detach_state()
        del state
        torch.cuda.empty_cache()

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = v / num_steps

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = (
                f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}_step_{self.ts_counts_per_stage[stage]}.pt"
            )
            torch.save(vis_data, save_vis_path)
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }

    def get_grad_info(self):
        grad_norms = [cast_to_numpy(p.grad.norm()) for n, p in self.model.named_parameters() if p.grad is not None]
        grad_norm = sum(grad_norms) / len(grad_norms)
        if self.do_debug:
            self.processed_data["grad_norm"].append(grad_norm)
            self.processed_data["grad_norms"].append(grad_norms)
        return grad_norms, grad_norm
