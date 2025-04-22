import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.models.encoders import get_encoders
from pose_tracking.trainer import Trainer
from pose_tracking.utils.common import cast_to_numpy, detach_and_cpu
from pose_tracking.utils.misc import is_empty
from torch import nn
from tqdm import tqdm


class RNDNet(nn.Module):
    def __init__(self, use_depth=False, out_dim=512):
        super().__init__()
        self.use_depth = use_depth
        self.out_dim = out_dim

        self.encoder_rgb, self.encoder_depth = get_encoders(
            model_name="resnet18", norm_layer_type="id", weights_rgb=None, weights_depth=None, out_dim=self.out_dim
        )
        self.target_rgb, self.target_depth = get_encoders(
            model_name="resnet18", norm_layer_type="id", weights_rgb=None, weights_depth=None, out_dim=self.out_dim
        )
        for param in self.target_rgb.parameters():
            param.requires_grad = False
        for param in self.target_depth.parameters():
            param.requires_grad = False

        if not self.use_depth:
            self.encoder_depth = None
            self.target_depth = None

    def forward(self, rgb, depth=None):
        res = {}
        with torch.no_grad():
            if self.use_depth:
                target_depth_out = self.target_depth(depth)
                res["target_depth_out"] = target_depth_out
            target_rgb_out = self.target_rgb(rgb)
            res["target_rgb_out"] = target_rgb_out

        if self.use_depth:
            depth_out = self.encoder_depth(depth)
            res["depth_out"] = depth_out
        rgb_out = self.encoder_rgb(rgb)
        res["rgb_out"] = rgb_out

        return res


class TrainerRND(Trainer):

    def __init__(
        self,
        *args_,
        args,
        **kwargs,
    ):
        self.encoder_module_prefix = "encoder"

        super().__init__(*args_, args=args, **kwargs)
        self.criterion = F.mse_loss

        self.use_pose = False

        params_wo_grad = [
            (i, n)
            for i, (n, p) in enumerate(self.model_without_ddp.named_parameters())
            if not p.requires_grad and "target" not in n
        ]
        if len(params_wo_grad):
            self.logger.warning(f"Params without grad: {params_wo_grad}")
            self.logger.warning(f"{len(params_wo_grad)=}")

        assert self.seq_len == 1

    def get_param_dicts(self):
        param_dicts = [
            {
                "params": [p for name, p in self.model_without_ddp.named_parameters() if "target" not in name],
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
        batch_size = len(batched_seq[0]["rgb"])
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
                vis_keys = ["rgb", "mask", "depth"]
                for k in vis_keys:
                    if len(batch_t.get(k, [])) == 0:
                        continue
                    vis_data[k].append(detach_and_cpu(batch_t[k]))
                vis_data["out"].append(detach_and_cpu(out))
            if self.do_debug:
                for k, v in batch_t.items():
                    if k not in ["rgb", "mask", "depth"] or is_empty(v):
                        continue
                    self.processed_data[k].append(v)
                for k, v in out.items():
                    if k not in ["rgb_out", "depth_out", "target_rgb_out", "target_depth_out"] or is_empty(v):
                        continue
                    self.processed_data[k].append(v)
                for k, v in {**loss_dict, **model_forward_res["metrics_dict"]}.items():
                    self.processed_data[k].append(v)

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

        out = self.model(
            image,
            depth=depth,
        )

        loss_rgb = self.criterion(out["rgb_out"], out["target_rgb_out"])
        loss = loss_rgb
        loss_dict = {"loss_rgb": loss_rgb}
        metrics_dict = {"cos_sim_rgb": F.cosine_similarity(out["rgb_out"], out["target_rgb_out"])}
        if self.model_without_ddp.use_depth:
            loss_depth = self.criterion(out["depth_out"], out["target_depth_out"])
            loss += loss_depth
            loss_dict["loss_depth"] = loss_depth
            metrics_dict["cos_sim_depth"] = F.cosine_similarity(out["depth_out"], out["target_depth_out"])
        loss_dict["loss"] = loss
        return {"out": out, "loss_dict": loss_dict, "metrics_dict": metrics_dict}
