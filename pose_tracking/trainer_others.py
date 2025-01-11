import math
import sys
from collections import defaultdict

import numpy as np
import torch
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.dataset.ds_common import from_numpy
from pose_tracking.dataset.pizza_utils import extend_seq_with_pizza_args
from pose_tracking.trainer import Trainer
from pose_tracking.utils.artifact_utils import save_results
from pose_tracking.utils.misc import reduce_metric
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from tqdm.auto import tqdm


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
            # batched_seq_forward
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

            if not math.isfinite(loss):
                self.logger(f"Loss is {loss}, stopping training")
                self.logger(f"{loss_z=} {loss_uv=} {R_metrics=} {T_metrics=}")
                sys.exit(1)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Call step of optimizer to update model params
                optimizer.step()
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(name)

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
                pose_mat_pred_abs = torch.stack([convert_pose_vector_to_matrix(rt) for rt in res["pose"]])
                save_results(batch, pose_mat_pred_abs, preds_dir)

        return running_stats
