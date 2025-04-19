import functools
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from cycler import K
from pose_tracking.config import TF_DIR, default_logger
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
from pose_tracking.models.encoders import (
    FrozenBatchNorm2d,
    get_encoders,
    is_param_part_of_encoders,
)
from pose_tracking.models.matcher import HungarianMatcher
from pose_tracking.models.set_criterion import SetCriterion
from pose_tracking.trainer import Trainer
from pose_tracking.utils.artifact_utils import save_results, save_results_v2
from pose_tracking.utils.common import cast_to_numpy, detach_and_cpu, extract_idxs
from pose_tracking.utils.detr_utils import postprocess_detr_outputs
from pose_tracking.utils.geom import (
    cam_to_2d,
    convert_2d_t_to_3d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
    rot_mat_from_6d,
    rotate_pts_batch,
)
from pose_tracking.utils.misc import (
    is_tensor,
    match_module_by_name,
    print_cls,
    reduce_dict,
    reduce_metric,
    split_arr,
)
from pose_tracking.utils.pipe_utils import get_trackformer_args
from pose_tracking.utils.pose import convert_pose_vector_to_matrix, convert_r_t_to_rt
from pose_tracking.utils.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class TrainerRND(Trainer):

    def __init__(
        self,
        *args_,
        args,
        opt_only=None,
        focal_alpha=0.25,
        kpt_spatial_dim=2,
        do_calibrate_kpt=False,
        use_pose_tokens=False,
        **kwargs,
    ):
        self.encoder_module_prefix = "backbone"

        super().__init__(*args_, args=args, **kwargs)
        self.criterion = F.mse_loss

        self.use_pose = False

        self.target_net = get_encoders(
            model_name="resnet18", norm_layer_type="id", weights_rgb=None, out_dim=self.args.encoder_out_dim
        )[0].to(self.device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        if self.use_ddp:
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        self.init_optimizer()

        params_wo_grad = [
            (i, n) for i, (n, p) in enumerate(self.model_without_ddp.named_parameters()) if not p.requires_grad
        ]
        if len(params_wo_grad):
            self.logger.warning(f"Params without grad: {params_wo_grad}")
            self.logger.warning(f"{len(params_wo_grad)=}")

        assert self.seq_len == 1

    def get_param_dicts(self):
        param_dicts = [
            {
                "params": [p for name, p in self.model_without_ddp.named_parameters()],
                "lr": self.args.lr,
            },
        ]

        return param_dicts

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

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["image"])
        if self.do_debug:
            for batch_t in batched_seq:
                for k, v in batch_t.items():
                    if k not in ["image", "intrinsics", "mesh_bbox", "bbox_2d", "class_id"]:
                        continue
                    self.processed_data[k].append(v)
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(list)
        seq_metrics = defaultdict(list)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq), disable=True)

        total_losses = []

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        failed_ts = []

        for t, batch_t in ts_pbar:
            model_forward_res = self.model_forward(batch_t)
            out = model_forward_res["out"]

            # POSTPROCESS OUTPUTS

            ...

            # LOSSES

            loss_dict = model_forward_res["loss_dict"]
            total_loss = loss_dict["loss"]

            if do_opt_every_ts:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if self.do_debug or do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    if do_vis:
                        vis_data["grad_norm"].append(grad_norm)
                        vis_data["grad_norms"].append(grad_norms)
                optimizer.step()
                optimizer.zero_grad()

            # METRICS

            for k, v in model_forward_res["metrics_dict"].items():
                seq_metrics[k].append(cast_to_numpy(v))

            # UPDATE VARS

            ...

            # OTHER

            loss_value = total_loss.item()
            seq_stats["loss"].append(loss_value)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert self.use_pose
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["image", "mesh_bbox", "mask", "depth"]
                for k in vis_keys:
                    if len(batch_t.get(k, [])) == 0:
                        continue
                    vis_data[k].append(detach_and_cpu(batch_t[k]))
                vis_data["out"].append(detach_and_cpu(out))

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = np.mean(v)

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

    def model_forward(self, batch_t, pose_tokens=None, prev_tokens=None, use_prev_image=False, **kwargs):

        image = batch_t["rgb"]
        depth = batch_t["depth"]

        with torch.no_grad():
            target = self.target_net(image)

        # extra_kwargs = {}
        # extra_kwargs["depth"] = depth

        out = self.model(
            image,
            # **extra_kwargs,
        )

        loss_dict = {"loss": self.criterion(out, target)}
        metrics_dict = {"cos_sim": F.cosine_similarity(out, target, dim=1).mean()}

        return {"out": out, "loss_dict": loss_dict, "metrics_dict": metrics_dict}
