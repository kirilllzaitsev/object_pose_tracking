import functools
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.dataset.ds_common import convert_batch_seq_to_seq_batch, from_numpy
from pose_tracking.dataset.pizza_utils import extend_seq_with_pizza_args
from pose_tracking.metrics import calc_metrics
from pose_tracking.trainer import Trainer
from pose_tracking.utils.artifact_utils import save_results
from pose_tracking.utils.common import detach_and_cpu
from pose_tracking.utils.geom import (
    convert_2d_t_to_3d,
    convert_3d_t_for_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
)
from pose_tracking.utils.misc import reduce_dict, reduce_metric, split_arr
from pose_tracking.utils.pose import convert_pose_vector_to_matrix, convert_r_t_to_rt
from pose_tracking.utils.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
)
from scipy.spatial.transform import Rotation
from tqdm import tqdm


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
    ):
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        seq_pbar = tqdm(loader, desc="Seq", leave=False, disable=len(loader) == 1)
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0 and stage == "train"

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
            running_stats[k] = v / len(loader)
        running_stats = reduce_dict(running_stats, device=self.device)

        if self.do_log:
            for k, v in running_stats.items():
                self.writer.add_scalar(f"{stage}_epoch/{k}", v, self.train_epoch_count)

        return running_stats

    def chunkify_batched_seq(self, batched_seq):
        num_chunks = len(batched_seq["rgb"][0]) // self.seq_len
        batched_seq_chunks = {
            k: ([split_arr(vv, num_chunks) if len(v) > 0 else v for vv in v]) for k, v in batched_seq.items()
        }
        batched_seq_chunks = {k: [vv for vv in zip(*v)] for k, v in batched_seq_chunks.items()}

        chunks = []
        for cidx in tqdm(range(num_chunks), desc="Subseq", leave=False):
            chunk = {k: v[cidx] for k, v in batched_seq_chunks.items()}
            for k in ["rgb", "pose", "center_depth"]:
                if k in chunk:
                    chunk[k] = torch.stack(chunk[k])
            chunks.append(chunk)
        return chunks

    def chunkify_batched_seq_sliding_window(self, batched_seq, window_size):
        num_chunks = len(batched_seq["rgb"][0]) - window_size + 1
        chunks = []
        for cidx in tqdm(range(num_chunks), desc="Subseq", leave=False):
            chunk = {
                k: (
                    v[:, cidx : cidx + window_size]
                    if isinstance(v, torch.Tensor)
                    else [vv[cidx : cidx + window_size] for vv in v]
                )
                for k, v in batched_seq.items()
            }
            chunks.append(chunk)
        return chunks

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
        do_opt_in_the_end = is_train

        seq_length = len(batched_seq["rgb"][0])
        batch_size = len(batched_seq["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)

        total_loss = 0

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        out_prev = None  # raw ouput of the model
        pose_mat_prev_gt_abs = None
        prev_latent = None
        state = None
        nan_count = 0
        do_skip_first_step = False

        if stage == "train" and len(batched_seq["rgb"]) == 1:
            self.logger.warning("Received bs=1 for training.")
            return {"losses": {}, "metrics": {}}
        if stage == "train":
            out_ts_raw = self.model(
                batched_seq["rgb"],
            )
        else:
            chunks = self.chunkify_batched_seq_sliding_window(batched_seq, window_size=self.seq_len)
            out_ts_raw = []
            for chunk_idx, chunk in enumerate(chunks):
                with torch.no_grad():
                    out_ts_raw_chunk = self.model(
                        chunk["rgb"],
                    )
                if chunk_idx > 0:
                    # the delta for previous timesteps is known from the last chunk
                    out_ts_raw_chunk = {k: v[:, -1:] for k, v in out_ts_raw_chunk.items()}
                out_ts_raw.append(out_ts_raw_chunk)
            # out_ts_raw is arr of dicts
            out_ts_raw = {
                k: torch.cat([out_ts_raw_chunk[k] for out_ts_raw_chunk in out_ts_raw], dim=1) for k in out_ts_raw[0]
            }
        # out_ts is k x seq_len-1 x bs
        out_ts = convert_batch_seq_to_seq_batch(out_ts_raw)

        seq_std = convert_batch_seq_to_seq_batch(batched_seq, device=self.device)
        ts_pbar = tqdm(
            enumerate(seq_std),
            desc="Timestep",
            leave=False,
            total=len(seq_std),
            disable=True,
        )

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
            t_gt_abs = pose_gt_abs[..., :3]
            rot_gt_abs = pose_gt_abs[..., 3:]

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

                    # prev_latent = torch.cat(
                    #     [self.model_without_ddp.encoder_img(rgb), self.model_without_ddp.encoder_depth(depth)], dim=1
                    # )

                    if save_preds:
                        assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                        save_results(batch_t, pose_mat_prev_gt_abs, preds_dir, gt_pose=pose_mat_prev_gt_abs)

                    do_skip_first_step = True
                    continue

            if self.use_prev_pose_condition:
                prev_pose = pose_prev_pred_abs if self.do_predict_rel_pose else out_prev
            else:
                prev_pose = None

            out = out_ts[t - 1]

            # POSTPROCESS OUTPUTS

            rot_pred, t_pred = out["rot"], out["t"]

            if self.do_predict_2d_t:
                center_depth_pred = out["center_depth"].squeeze(-1)
                convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(t_pred, center_depth_pred, intrinsics.float(), hw=hw)
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
                        if self.do_predict_3d_rot:
                            rot_gt_rel = matrix_to_axis_angle(rot_gt_rel_mat)
                        elif self.do_predict_6d_rot:
                            rot_gt_rel = matrix_to_rotation_6d(rot_gt_rel_mat)
                        else:
                            rot_gt_rel = matrix_to_quaternion(rot_gt_rel_mat)
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_rel)
                else:
                    if self.use_rot_mat_for_loss:
                        loss_rot = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_abs)

                if self.include_abs_pose_loss_for_rel:
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

            total_loss += loss

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
                seq_metrics[k] += v

            # OTHER

            seq_stats["loss"] += loss.item()
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose.item()
            else:
                seq_stats["loss_rot"] += loss_rot.item()
                seq_stats["loss_t"] += loss_t.item()
                if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                    seq_stats["loss_t_abs"] += loss_t_abs.item()
                    seq_stats["loss_rot_abs"] += loss_rot_abs.item()

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
            pose_prev_pred_abs = {k: v for k, v in pose_prev_pred_abs.items()}

            pose_mat_prev_gt_abs = pose_mat_gt_abs
            out_prev = {"t": out["t"], "rot": out["rot"]}

        num_steps = seq_length
        if do_skip_first_step:
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

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = v / num_steps

        if self.do_predict_rel_pose:
            # calc loss/metrics btw accumulated abs poses
            metrics_abs = self.calc_metrics_batch(batch_t, pose_mat_pred_abs, pose_mat_gt_abs)
            for k, v in metrics_abs.items():
                seq_metrics[f"{k}_abs"] += v
            if not self.include_abs_pose_loss_for_rel:
                with torch.no_grad():
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    seq_stats["loss_t_abs"] = loss_t_abs.item()
                    seq_stats["loss_rot_abs"] = loss_rot_abs.item()

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
