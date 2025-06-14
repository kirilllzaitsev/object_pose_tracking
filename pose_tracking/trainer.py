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
from pose_tracking.dataset.transforms import mask_pixels, mask_pixels_torch
from pose_tracking.losses import compute_chamfer_dist, kpt_cross_ratio_loss, silog_loss
from pose_tracking.metrics import (
    calc_metrics,
    calc_r_error,
    calc_rt_errors,
    calc_t_error,
    eval_batch_det,
)
from pose_tracking.models.encoders import FrozenBatchNorm2d, is_param_part_of_encoders
from pose_tracking.models.matcher import HungarianMatcher
from pose_tracking.models.set_criterion import SetCriterion
from pose_tracking.utils.artifact_utils import save_results, save_results_v2
from pose_tracking.utils.common import cast_to_numpy, detach_and_cpu, extract_idxs
from pose_tracking.utils.detr_utils import postprocess_detr_outputs
from pose_tracking.utils.geom import (
    backproj_2d_pts,
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
)
from pose_tracking.utils.misc import (
    distributed_rank,
    is_empty,
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
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        device,
        hidden_dim,
        rnn_type,
        seq_len,
        args,
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
        use_belief_decoder=True,
        use_priv_decoder=False,
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
        model_name="",
        opt_only=None,
        max_clip_grad_norm=0.1,
        tf_t_loss_coef=1,
        tf_rot_loss_coef=1,
        use_entire_seq_in_train=False,
        use_seq_len_curriculum=False,
        use_pnp_for_rot_pred=False,
        do_predict_abs_pose=False,
        use_m_for_metrics=True,
        use_factors=False,
        seq_len_max=None,
        seq_len_curriculum_step_epoch_freq=10,
        bbox_num_kpts=32,
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
        self.use_belief_decoder = use_belief_decoder
        self.use_priv_decoder = use_priv_decoder
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
        self.use_seq_len_curriculum = use_seq_len_curriculum
        self.do_predict_abs_pose = do_predict_abs_pose
        self.use_pnp_for_rot_pred = use_pnp_for_rot_pred
        self.use_factors = use_factors
        self.use_m_for_metrics = use_m_for_metrics

        self.args = args
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
        self.bbox_num_kpts = bbox_num_kpts
        if use_seq_len_curriculum:
            self.seq_len_init = 2 if do_predict_rel_pose else 1
            self.seq_len_current = self.seq_len_init
            if seq_len_max is None:
                self.seq_len_max = seq_len
            else:
                self.seq_len_max = seq_len_max
            self.seq_len_curriculum_step_epoch_freq = seq_len_curriculum_step_epoch_freq
        else:
            self.seq_len_max = None
            self.seq_len_curriculum_step_epoch_freq = None

        if self.use_ddp:
            self.model = DDP(self.model)
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        self.do_reset_state = "lstm" in model_name
        self.use_pose_loss = criterion_pose is not None
        self.include_abs_pose_loss_for_rel = include_abs_pose_loss_for_rel and do_predict_rel_pose
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn
        self.vis_dir = f"{self.exp_dir}/vis"
        self.use_rot_mat_for_loss = self.criterion_rot_name in [
            "displacement",
            "geodesic_mat",
            "geodesic_mat_sym",
            "adds",
        ]
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
        if criterion_rot_name in ["geodesic"]:
            assert not (do_predict_3d_rot or do_predict_6d_rot)
        if use_pnp_for_rot_pred:
            assert self.do_predict_kpts

        self.init_optimizer()

    def init_optimizer(self):
        param_dicts = self.get_param_dicts()

        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

    def __repr__(self):
        return print_cls(self, excluded_attrs=["processed_data", "model", "args", "model_without_ddp"])

    def get_param_dicts(self):
        param_dicts = [
            {
                "params": [
                    p
                    for name, p in self.model_without_ddp.named_parameters()
                    if is_param_part_of_encoders(name, encoder_module_prefix="encoder")
                ],
                "lr": self.args.lr_encoders,
            },
            {
                "params": [
                    p
                    for name, p in self.model_without_ddp.named_parameters()
                    if not is_param_part_of_encoders(name, encoder_module_prefix="encoder")
                ],
                "lr": self.args.lr,
            },
        ]
        return param_dicts

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
            self.model.train()
        else:
            self.model.eval()
        running_stats = defaultdict(list)
        seq_pbar = tqdm(loader, desc="Seq", leave=False, disable=len(loader) == 1)
        do_vis = self.do_vis and (self.train_epoch_count) % self.vis_epoch_freq == 0 and stage in ["train"]

        if self.use_seq_len_curriculum:
            if self.train_epoch_count % self.seq_len_curriculum_step_epoch_freq == 0:
                self.seq_len_current = min(self.seq_len_current + 1, self.seq_len_max)
            seq_len = self.seq_len_current
        else:
            seq_len = self.seq_len

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):

            if is_empty(batched_seq):
                self.logger.warning(f"Empty batch at seq_pack_idx {seq_pack_idx}, skipping...")
                continue

            if self.do_reset_state:
                batch_size = len(batched_seq[0]["rgb"])
                self.model_without_ddp.reset_state(batch_size, device=self.device)

            if (stage != "train" and not self.do_chunkify_val) or (
                stage == "train" and not self.use_entire_seq_in_train
            ):
                seq_stats = self.batched_seq_forward(
                    batched_seq=batched_seq,
                    optimizer=optimizer,
                    save_preds=save_preds,
                    preds_dir=preds_dir,
                    stage=stage,
                    do_vis=do_vis,
                )
            else:
                batched_seq_chunks = split_arr(batched_seq, len(batched_seq) // seq_len)
                seq_stats = defaultdict(lambda: defaultdict(list))
                num_chunks = len(batched_seq_chunks)
                last_step_state = None

                for cidx, chunk in tqdm(
                    enumerate(batched_seq_chunks), desc="Subseq", leave=False, total=num_chunks, disable=True
                ):
                    seq_stats_chunk = self.batched_seq_forward(
                        batched_seq=chunk,
                        optimizer=optimizer,
                        save_preds=save_preds,
                        preds_dir=preds_dir,
                        stage=stage,
                        do_vis=do_vis,
                        last_step_state=last_step_state,
                    )
                    last_step_state = seq_stats_chunk.pop("last_step_state")
                    for k, v in seq_stats_chunk.items():
                        for kk, vv in v.items():
                            seq_stats[k][kk].append(vv)
                    if cidx == 1:
                        do_vis = False
            torch.cuda.empty_cache()

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                running_stats[k].append(v)
                if self.do_log and self.do_log_every_seq:
                    self.writer.add_scalar(f"{stage}_seq/{k}", np.mean(v), self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            if self.do_print_seq_stats:
                seq_pbar.set_postfix({k: np.mean(v) for k, v in running_stats.items()})

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            running_stats[k] = np.mean(v)
        running_stats = reduce_dict(running_stats, device=self.device)

        if self.do_debug:
            running_stats["full"] = seq_stats

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
        depth_no_noise = None

        for t, batch_t in ts_pbar:
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"].squeeze(1)
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            features_rgb = batch_t.get("features_rgb")
            h, w = rgb.shape[-2:]
            hw = (h, w)
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]
            depth_no_noise = depth if isinstance(depth, list) else depth.clone()
            if self.args.mask_pixels_prob > 0 and is_train:
                depth = mask_pixels_torch(
                    depth, p=self.args.mask_pixels_prob, pixels_masked_max_percent=0.1, use_noise=True, use_blocks=True
                )

            if self.do_predict_rel_pose:
                if t == 0 and last_step_state is not None:
                    pose_mat_prev_gt_abs = last_step_state["pose_mat_prev_gt_abs"]
                    pose_prev_pred_abs = last_step_state["pose_prev_pred_abs"]
                    prev_latent = last_step_state.get("prev_latent")
                    state = last_step_state.get("state")
                elif t == 0:
                    if self.do_perturb_init_gt_for_rel_pose and is_train:
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

                    pose_mat_prev_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])

                    out = self.model(
                        rgb=rgb,
                        depth=depth,
                        bbox=bbox_2d,
                        state=None,
                        features_rgb=features_rgb,
                        bbox_kpts=batch_t[kpts_key],
                        depth_no_noise=depth_no_noise,
                    )
                    state = out["state"]

                    prev_latent = out["latent"] if self.use_prev_latent else None

                    if self.do_predict_abs_pose:
                        t_abs_pose, rot_abs_pose = out["t_abs_pose"], out["rot_abs_pose"]
                        losses_abs_pose = self.calc_abs_pose_loss_for_rel(
                            pose_gt_abs, t_pred_abs_pose=t_abs_pose, rot_pred_abs_pose=rot_abs_pose
                        )
                        total_losses.append(losses_abs_pose["loss_abs_pose"])

                        pose_prev_pred_abs = {"t": t_abs_pose, "rot": rot_abs_pose}

                        for k, v in losses_abs_pose.items():
                            seq_stats[k].append(v.item())
                    else:
                        pose_prev_pred_abs = {"t": t_gt_abs + noise_t, "rot": rot_gt_abs}

                    if self.use_pnp_for_rot_pred:
                        prev_kpts = out["kpts"]

                    if save_preds:
                        assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                        save_results(batch_t, pose_mat_prev_gt_abs, preds_dir, gt_pose=pose_mat_prev_gt_abs)

                    # kpt loss
                    if self.do_predict_kpts:
                        loss = self.calc_kpt_loss(batch_t, out)["loss"]
                        total_losses.append(loss)

                    if do_vis:
                        self.extend_vis_data_with_batch_t(vis_batch_idxs, vis_data, batch_t, out=out)

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
                bbox_kpts_prev=batched_seq[t - 1][kpts_key] if t > 0 else None,
                depth_no_noise=depth_no_noise,
            )

            # POSTPROCESS OUTPUTS

            rot_pred, t_pred = out["rot"], out["t"]
            state = out["state"]

            if self.do_predict_2d_t:
                if self.do_predict_rel_pose:
                    scale_pred = out["center_depth"].squeeze(-1)
                    center_depth_pred = ((scale_pred) + 1) * pose_prev_pred_abs["t"][..., 2]
                    t_pred = torch.zeros_like(t_gt_abs)
                    t_pred[..., :2] = out["t"]
                    t_pred[..., 2] = center_depth_pred - pose_prev_pred_abs["t"][..., 2]
                else:
                    center_depth_pred = out["center_depth"].squeeze(1)
                    convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                        t_pred, center_depth_pred, intrinsics.float(), hw=hw
                    )
                    t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

            pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])
            rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]

            pose_mat_pred = torch.stack(
                [self.pose_to_mat_converter_fn(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
            )

            if self.use_pnp_for_rot_pred:
                kpts = out["kpts"]
                rot_mat_pred_bidxs = []
                t_mat_pred_bidxs = []
                for bidx in range(batch_size):
                    kpts_2d_denorm = kpts[bidx] * torch.tensor([w, h]).to(self.device)
                    if self.do_predict_rel_pose:
                        prev_kpts_2d = prev_kpts[bidx]
                        prev_kpts_2d_denorm = prev_kpts_2d * torch.tensor([w, h]).to(self.device)
                        prev_depth = batched_seq[t - 1]["depth"][bidx]
                        prev_kpts_depth_actual = prev_depth[
                            ..., prev_kpts_2d_denorm[:, 1].long(), prev_kpts_2d_denorm[:, 0].long()
                        ].double()
                        prev_kpts_depth = batched_seq[t - 1]["bbox_2d_kpts_depth"][bidx] / 10
                        prev_visib_kpt_mask = torch.abs(prev_kpts_depth - prev_kpts_depth_actual) < 1e-2
                        prev_kpts_2d_denorm_visib = prev_kpts_2d_denorm[prev_visib_kpt_mask]
                        prev_kpts_visib_depth = prev_kpts_depth[prev_visib_kpt_mask]
                        prev_kpts_3d = backproj_2d_pts(
                            prev_kpts_2d_denorm_visib, prev_kpts_visib_depth, intrinsics[bidx]
                        )
                        visib_kpt_mask = prev_visib_kpt_mask
                        kpts_3d = prev_kpts_3d
                    else:
                        visib_kpt_mask = slice(None)
                        kpts_3d = batch_t["bbox_3d_kpts_mesh"][bidx]
                    kpts_2d_denorm_visib = kpts_2d_denorm[visib_kpt_mask]
                    pose_from_3d_2d_matches_res = get_pose_from_3d_2d_matches(
                        kpts_3d, kpts_2d_denorm_visib, intrinsics[bidx]
                    )
                    rot_mat_pred_bidx = pose_from_3d_2d_matches_res["R"]
                    t_mat_pred_bidx = pose_from_3d_2d_matches_res["t"]
                    rot_mat_pred_bidxs.append(torch.tensor(rot_mat_pred_bidx))
                    t_mat_pred_bidxs.append(torch.tensor(t_mat_pred_bidx))

                rot_mat_pred = torch.stack(rot_mat_pred_bidxs).to(self.device)
                rot_pred = self.rot_mat_to_vector_converter_fn(rot_mat_pred)
                t_pred = torch.stack(t_mat_pred_bidxs).to(self.device).squeeze(-1)
                pose_mat_pred[:, :3, :3] = rot_mat_pred
                pose_mat_pred[:, :3, 3] = t_pred

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
                        t_pred_2d = out["t"]
                        loss_uv = self.criterion_trans(t_pred_2d[..., :2], t_gt_rel[..., :2]) * (1e0)
                        scale_gt = t_gt_abs[..., 2] / pose_mat_prev_gt_abs[..., 2, 3] - 1
                        loss_z = self.criterion_trans(scale_pred, (scale_gt)) * (1e0)
                    else:
                        t_gt_2d_norm, depth_gt = convert_3d_t_for_2d(t_gt_abs, intrinsics, hw)
                        loss_uv = self.criterion_trans(t_pred_2d, t_gt_2d_norm.squeeze(1))
                        loss_z = self.criterion_trans(center_depth_pred, depth_gt)
                    loss_t = loss_uv + loss_z
                else:
                    if self.do_predict_rel_pose:
                        rel_t_scaler = 1
                        loss_t = self.criterion_trans(t_pred * rel_t_scaler, t_gt_rel * rel_t_scaler)
                    else:
                        loss_t = self.criterion_trans(t_pred_abs, t_gt_abs)

                # rot loss

                if self.criterion_rot_name in ["displacement", "adds"]:
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
                    assert any(x in self.opt_only for x in ["rot", "t"]), f"Invalid opt_only: {self.opt_only}"
                    if "rot" in self.opt_only:
                        loss += loss_rot * self.tf_rot_loss_coef
                    if "t" in self.opt_only:
                        loss += loss_t * self.tf_t_loss_coef

            # depth loss
            if self.use_belief_decoder:
                loss_depth = F.mse_loss(out["decoder_out"]["depth_final"], out["depth_no_noise_latent"])
                loss_depth_rec = silog_loss(out["depth_no_noise_rec"], depth_no_noise)
                loss += loss_depth + loss_depth_rec

            # priv loss
            if self.use_priv_decoder:
                loss_priv = 0.01 * F.mse_loss(out["priv_decoded"], batch_t["bbox_3d_kpts"])
                loss += loss_priv

            # kpt loss
            if self.do_predict_kpts:
                calc_kpt_loss_res = self.calc_kpt_loss(batch_t, out)
                loss_kpts = calc_kpt_loss_res["loss_kpts"]
                loss_cr = calc_kpt_loss_res["loss_cr"]
                loss += calc_kpt_loss_res["loss"]

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
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                optimizer.step()
                optimizer.zero_grad()
                state = [x if x is None else x.detach() for x in state]
            elif do_opt_in_the_end:
                total_losses.append(loss)

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
            if self.use_belief_decoder:
                seq_stats["loss_depth"].append(loss_depth.item())
                seq_stats["loss_depth_rec"].append(loss_depth_rec.item())
            if self.use_priv_decoder:
                seq_stats["loss_priv"].append(loss_priv.item())
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
            if self.do_predict_kpts:
                seq_stats["loss_kpts"].append(loss_kpts.item())
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
                self.extend_vis_data_with_batch_t(vis_batch_idxs, vis_data, batch_t, out=out)
                vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs[vis_batch_idxs]))
                vis_data["pose_mat_pred"].append(detach_and_cpu(pose_mat_pred[vis_batch_idxs]))
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

        if do_opt_in_the_end:
            total_loss = torch.mean(torch.stack(total_losses))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                vis_data["grad_norm"].append(grad_norm)
                vis_data["grad_norms"].append(grad_norms)
            optimizer.step()
            optimizer.zero_grad()

        if self.use_rnn:
            last_step_state = {
                "prev_latent": prev_latent.detach() if self.use_prev_latent else None,
                "pose_prev_pred_abs": {k: v.detach() for k, v in pose_prev_pred_abs.items()},
                "pose_mat_prev_gt_abs": pose_mat_prev_gt_abs,
                "state": [
                    x if x is None else ((xx.detach() for xx in x) if isinstance(x, tuple) else x.detach())
                    for x in state
                ],
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
            torch.save(vis_data, save_vis_path)
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
            "last_step_state": last_step_state,
        }

    def extend_vis_data_with_batch_t(self, vis_batch_idxs, vis_data, batch_t, out=None):
        vis_keys = ["rgb", "intrinsics", "pose"]
        for k in ["mask", "mesh_bbox", "pts", "depth", "bbox_2d_kpts"]:
            if k in batch_t and len(batch_t[k]) > 0:
                vis_keys.append(k)
        for k in vis_keys:
            vis_data[k].append([detach_and_cpu(batch_t[k][i]) for i in vis_batch_idxs])
        if out is not None:
            vis_data["out"].append(detach_and_cpu(out))

    def calc_kpt_loss(self, batch_t, out):
        kpts_pred = out["kpts"]
        kpts_gt = batch_t["bbox_2d_kpts"].float()
        loss_kpts = 1e1 * F.huber_loss(kpts_pred, kpts_gt)
        loss = loss_kpts
        # bbox_2d_kpts_collinear_idxs = batch_t["bbox_2d_kpts_collinear_idxs"]
        # loss_cr = kpt_cross_ratio_loss(kpts_pred, bbox_2d_kpts_collinear_idxs)
        loss_cr = torch.tensor(0.0, device=loss_kpts.device)
        loss += loss_cr * 0.01
        return {
            "loss_kpts": loss_kpts,
            "loss_cr": loss_cr,
            "loss": loss,
        }

    def calc_metrics_batch(self, batch_t, pose_mat_pred_metrics, pose_mat_gt_metrics):
        bbox_3d = batch_t["mesh_bbox"]
        diameter = batch_t["mesh_diameter"]
        pts = batch_t["mesh_pts"]
        m_batch = defaultdict(list)
        for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_mat_pred_metrics, pose_mat_gt_metrics)):
            m_sample = calc_metrics(
                pred_rt=pred_rt,
                gt_rt=gt_rt,
                pts=pts[sample_idx],
                class_name=None,
                use_miou=True,
                bbox_3d=bbox_3d[sample_idx],
                diameter=diameter[sample_idx],
                is_meters=True,
                use_m=self.use_m_for_metrics,
                log_fn=print if self.logger is None else self.logger.warning,
            )
            for k, v in m_sample.items():
                m_batch[k].append(v)

        m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
        return m_batch_avg

    def calc_abs_pose_loss_for_rel(self, pose_gt_abs, t_pred_abs_pose, rot_pred_abs_pose):
        t_gt_abs = pose_gt_abs[:, :3]
        rot_gt_abs = pose_gt_abs[:, 3:]
        loss_rot_abs_pose = self.criterion_rot(rot_pred_abs_pose, rot_gt_abs)
        loss_t_abs_pose = self.criterion_trans(t_pred_abs_pose, t_gt_abs)
        loss = loss_rot_abs_pose + loss_t_abs_pose
        return {
            "loss_rot_abs_pose": loss_rot_abs_pose,
            "loss_t_abs_pose": loss_t_abs_pose,
            "loss_abs_pose": loss,
        }

    def get_grad_info(self):
        grad_norms = [(n, p.grad.norm().item()) for n, p in self.model.named_parameters() if p.grad is not None]
        grad_norm = sum([x[1] for x in grad_norms]) / max(len(grad_norms), 1)
        if self.do_debug:
            self.processed_data["grad_norm"].append(grad_norm)
            self.processed_data["grad_norms"].append(grad_norms)
        return grad_norms, grad_norm
