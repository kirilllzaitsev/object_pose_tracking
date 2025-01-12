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
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)
from tqdm.auto import tqdm


class TrainerDeformableDETR(Trainer):

    def __init__(
        self,
        *args,
        num_classes,
        aux_loss,
        num_dec_layers,
        focal_alpha=0.25,
        kpt_spatial_dim=2,
        opt_only=None,
        do_calibrate_kpt=False,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.do_calibrate_kpt = do_calibrate_kpt

        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.num_dec_layers = num_dec_layers
        self.kpt_spatial_dim = kpt_spatial_dim
        self.focal_alpha = focal_alpha

        self.cost_class, self.cost_bbox, self.cost_giou = (2, 5, 2)
        self.losses = [
            "labels",
            "boxes",
            "rot",
            "t",
        ]
        if opt_only is not None:
            self.losses = [v for v in self.losses if v in opt_only]
            if "labels" not in opt_only:
                self.cost_class = 0
            if "boxes" not in opt_only:
                self.cost_bbox = 0
                self.cost_giou = 0

        self.matcher = HungarianMatcher(cost_class=self.cost_class, cost_bbox=self.cost_bbox, cost_giou=self.cost_giou)
        self.weight_dict = {
            "loss_ce": 1,
            "loss_rot": 1,
            "loss_t": 1,
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
        self.losses += ["cardinality"]
        self.criterion = SetCriterion(
            num_classes - 1,
            self.matcher,
            self.weight_dict,
            self.losses,
            focal_alpha=focal_alpha,
        )

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
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0
        seq_pbar = tqdm(loader, desc="Seq", leave=False, disable=len(loader) == 1)

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
        if self.do_debug:
            for batch_t in batched_seq:
                for k, v in batch_t.items():
                    if k not in ["rgb", "intrinsics", "mesh_bbox", "bbox_2d", "class_id"]:
                        continue
                    self.processed_data[k].append(v)
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(
            enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq), disable=seq_length == 1
        )

        if do_opt_in_the_end:
            optimizer.zero_grad()
            total_loss = 0

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
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
                "rot": [v.float() for v in [v if v.ndim > 1 else v[None] for v in rot_gt_abs]],
                "t": [v.float() for v in [v if v.ndim > 1 else v[None] for v in t_gt_abs]],
                "masks": mask,
            }
            if self.model_name == "detr_kpt":
                extra_kwargs = {}
                if self.do_calibrate_kpt or self.kpt_spatial_dim > 2:
                    extra_kwargs["intrinsics"] = intrinsics
                if self.kpt_spatial_dim > 2:
                    extra_kwargs["depth"] = depth
                out = self.model(
                    rgb,
                    mask=mask,
                    **extra_kwargs,
                )
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if self.do_debug or do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    if do_vis:
                        vis_data["grad_norm"].append(grad_norm)
                        vis_data["grad_norms"].append(grad_norms)
                    if self.do_debug:
                        self.processed_data["grad_norm"].append(grad_norm)
                        self.processed_data["grad_norms"].append(grad_norms)

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
            for k in ["class_error"]:
                if k in loss_dict_reduced:
                    seq_stats[k] += loss_dict_reduced[k]

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
                for k in ["mask", "mesh_bbox", "pts", "class_id", "bbox_2d"]:
                    if k in batch_t and len(batch_t[k]) > 0:
                        vis_keys.append(k)
                for k in vis_keys:
                    vis_data[k].append([batch_t[k][i].cpu() for i in vis_batch_idxs])
                vis_data["targets"].append(extract_idxs(targets, vis_batch_idxs))
                vis_data["out"].append(extract_idxs(out, vis_batch_idxs, do_extract_dict_contents=True))
                if self.model_name == "detr_kpt":
                    vis_data["kpts"].append(extract_idxs(out["kpts"], vis_batch_idxs))
                    vis_data["descriptors"].append(extract_idxs(out["descriptors"], vis_batch_idxs))

                # vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

            if self.do_debug:
                # add everything to processed_data
                # self.processed_data["pose_gt_abs"].append(pose_gt_abs)
                # self.processed_data["pose_mat_gt_abs"].append(pose_mat_gt_abs)
                # self.processed_data["pose_mat_pred_abs"].append(pose_mat_pred_abs)
                # self.processed_data["pose_prev_pred_abs"].append(pose_prev_pred_abs)
                self.processed_data["targets"].append(detach_and_cpu(targets))
                # self.processed_data["m_batch"].append(detach_and_cpu(m_batch))
                # self.processed_data["out_prev"].append(detach_and_cpu(out_prev))
                self.processed_data["out"].append(detach_and_cpu(out))
                # self.processed_data["pred_classes"].append(detach_and_cpu(out["pred_logits"].argmax(-1) + 1))
                # self.processed_data["rot_pred"].append(rot_pred)
                # self.processed_data["t_pred"].append(t_pred)

        if do_opt_in_the_end:
            total_loss /= seq_length
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if self.do_debug or do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                if do_vis:
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                if self.do_debug:
                    self.processed_data["grad_norm"].append(grad_norm)
                    self.processed_data["grad_norms"].append(grad_norms)
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
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }


class TrainerTrackformer(Trainer):

    def __init__(
        self,
        *args_,
        args,
        num_classes,
        aux_loss,
        num_dec_layers,
        focal_alpha=0.25,
        kpt_spatial_dim=2,
        opt_only=None,
        do_calibrate_kpt=False,
        **kwargs,
    ):

        from pose_tracking.utils.pipe_utils import get_trackformer_args
        from trackformer.models import build_criterion
        from trackformer.models.matcher import build_matcher

        super().__init__(*args_, **kwargs)

        self.use_pose = opt_only is None or ("rot" in opt_only and "t" in opt_only)
        self.do_calibrate_kpt = do_calibrate_kpt
        self.tf_args = get_trackformer_args(args)
        self.args = args

        self.opt_only = opt_only
        self.num_classes = num_classes  # includes bg
        self.aux_loss = aux_loss
        self.num_dec_layers = num_dec_layers
        self.kpt_spatial_dim = kpt_spatial_dim
        self.focal_alpha = focal_alpha

        self.save_vis_paths = []

        self.matcher = build_matcher(self.tf_args)
        self.criterion = build_criterion(
            self.tf_args, num_classes=num_classes, matcher=self.matcher, device=args.device
        )

        if self.use_ddp:
            model_without_ddp = self.model.module
        else:
            model_without_ddp = self.model

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if not match_module_by_name(
                        n,
                        args.detr_args.lr_backbone_names
                        + args.detr_args.lr_linear_proj_names
                        + ["layers_track_attention"],
                    )
                    and p.requires_grad
                ],
                "lr": args.detr_args.lr,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if match_module_by_name(n, args.detr_args.lr_backbone_names) and p.requires_grad
                ],
                "lr": args.detr_args.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in model_without_ddp.named_parameters()
                    if match_module_by_name(n, args.detr_args.lr_linear_proj_names) and p.requires_grad
                ],
                "lr": args.detr_args.lr * args.detr_args.lr_linear_proj_mult,
            },
        ]
        if args.detr_args.track_attention:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in model_without_ddp.named_parameters()
                        if match_module_by_name(n, ["layers_track_attention"]) and p.requires_grad
                    ],
                    "lr": args.detr_args.lr_track,
                }
            )

        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=args.detr_args.lr,
            weight_decay=args.weight_decay,
        )

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
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0
        seq_pbar = tqdm(loader, desc="Seq", leave=False, disable=len(loader) == 1)
        if stage == "train":
            optimizer = self.optimizer

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
        batch_size = len(batched_seq[0]["image"])
        if self.do_debug:
            for batch_t in batched_seq:
                for k, v in batch_t.items():
                    if k not in ["image", "intrinsics", "mesh_bbox", "bbox_2d", "class_id"]:
                        continue
                    self.processed_data[k].append(v)
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(
            enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq), disable=seq_length == 1
        )

        if do_opt_in_the_end:
            optimizer.zero_grad()
            total_loss = 0

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        out_prev = None  # raw ouput of the model
        pose_mat_prev_gt_abs = None
        prev_latent = None
        nan_count = 0

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["image"]
            mask = batch_t["mask"]
            targets = batch_t["target"]
            pose_gt_abs = torch.stack([x["pose"] for x in targets])
            intrinsics = [x["intrinsics"] for x in targets]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            h, w = rgb.shape[-2:]
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            out, targets_res, *_ = self.model(rgb, targets)

            # POSTPROCESS OUTPUTS

            # LOSSES

            loss_dict = self.criterion(out, targets_res)
            indices = loss_dict.pop("indices")
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if self.do_debug or do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    if do_vis:
                        vis_data["grad_norm"].append(grad_norm)
                        vis_data["grad_norms"].append(grad_norms)
                    if self.do_debug:
                        self.processed_data["grad_norm"].append(grad_norm)
                        self.processed_data["grad_norms"].append(grad_norms)

                optimizer.step()
            elif do_opt_in_the_end:
                total_loss += loss

            # METRICS

            bbox_3d = batch_t["mesh_bbox"]
            diameter = batch_t["mesh_diameter"]
            m_batch = defaultdict(list)

            if self.use_pose:
                idx = self.criterion._get_src_permutation_idx(indices)
                target_rts = torch.cat(
                    [torch.cat([t["t"][i], t["rot"][i]], dim=1) for t, (_, i) in zip(targets, indices)], dim=0
                )

                pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in target_rts])
                t_pred = out["t"][idx]

                if self.do_predict_2d_t:
                    center_depth_pred = out["center_depth"][idx]
                    convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                        t_pred, center_depth_pred, intrinsics, hw=(h, w), do_predict_rel_pose=self.do_predict_rel_pose
                    )
                    t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

                rot_pred = out["rot"][idx]
                pred_rts = torch.cat([t_pred, rot_pred], dim=1)
                pose_mat_pred_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pred_rts])

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
            target_sizes = torch.stack([x["size"] for x in batch_t["target"]])
            out_formatted = postprocess_detr_outputs(out, target_sizes=target_sizes)
            m_batch_avg.update(eval_batch_det(out_formatted, targets, num_classes=self.num_classes - 1))

            for k, v in m_batch_avg.items():
                if "classes" in k:
                    continue
                seq_metrics[k] += v

            # UPDATE VARS

            # OTHER

            loss_value = losses_reduced_scaled.item()
            seq_stats["loss"] += loss_value
            for k, v in {**loss_dict_reduced_scaled}.items():
                if "indices" in k:
                    continue
                seq_stats[k] += v
            for k in ["class_error"]:
                if k in loss_dict_reduced:
                    seq_stats[k] += loss_dict_reduced[k]

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])

            if not math.isfinite(loss_value):
                self.logger(f"Loss is {loss_value}, stopping training")
                self.logger(loss_dict_reduced)
                sys.exit(1)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert self.use_pose
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
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
                    pose_gt=pose_gt_abs,
                    pose_pred=pose_mat_pred_abs,
                    rgb_path=batch_t["rgb_path"],
                    preds_dir=preds_dir,
                    bboxs=bboxs,
                    labels=labels,
                )

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["image", "mesh_bbox"]
                for k in vis_keys:
                    vis_data[k].append([batch_t[k][i].cpu() for i in vis_batch_idxs])
                vis_data["targets"].append(extract_idxs(targets, vis_batch_idxs))
                vis_data["out"].append(extract_idxs(out, vis_batch_idxs, do_extract_dict_contents=True))

                # vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

            if self.do_debug:
                # add everything to processed_data
                # self.processed_data["pose_gt_abs"].append(pose_gt_abs)
                # self.processed_data["pose_mat_gt_abs"].append(pose_mat_gt_abs)
                # self.processed_data["pose_mat_pred_abs"].append(pose_mat_pred_abs)
                # self.processed_data["pose_prev_pred_abs"].append(pose_prev_pred_abs)
                self.processed_data["targets"].append(detach_and_cpu(targets))
                # self.processed_data["m_batch"].append(detach_and_cpu(m_batch))
                # self.processed_data["out_prev"].append(detach_and_cpu(out_prev))
                self.processed_data["out"].append(detach_and_cpu(out))
                # self.processed_data["pred_classes"].append(detach_and_cpu(out["pred_logits"].argmax(-1) + 1))
                # self.processed_data["rot_pred"].append(rot_pred)
                # self.processed_data["t_pred"].append(t_pred)

        if do_opt_in_the_end:
            total_loss /= seq_length
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if self.do_debug or do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                if do_vis:
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                if self.do_debug:
                    self.processed_data["grad_norm"].append(grad_norm)
                    self.processed_data["grad_norms"].append(grad_norms)
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
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }
