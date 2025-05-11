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
from pose_tracking.models.encoders import FrozenBatchNorm2d, is_param_part_of_encoders
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
from pose_tracking.utils.kpt_utils import sample_gt_kpts
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
from pose_tracking.utils.segm_utils import mask_morph
from tqdm import tqdm


class TrainerDeformableDETR(Trainer):

    def __init__(
        self,
        *args_,
        model,
        num_classes,
        aux_loss,
        num_dec_layers,
        args,
        opt_only=None,
        focal_alpha=0.25,
        kpt_spatial_dim=2,
        do_calibrate_kpt=False,
        use_pose_tokens=False,
        use_kpt_loss=False,
        **kwargs,
    ):
        if "detr_kpt" in args.model_name:
            self.encoder_module_prefix = "extractor"
        elif any(x in args.model_name for x in ["detr", "trackformer", "memotr"]):
            self.encoder_module_prefix = "backbone"
        else:
            self.encoder_module_prefix = None

        if args.do_freeze_encoders:
            self.freeze_encoder(model_without_ddp=model)

        super().__init__(*args_, model=model, args=args, **kwargs)

        self.do_calibrate_kpt = do_calibrate_kpt
        self.use_pose_tokens = use_pose_tokens
        self.use_kpt_loss = use_kpt_loss

        self.num_classes = num_classes  # excluding bg class
        self.aux_loss = aux_loss
        self.num_dec_layers = num_dec_layers
        self.kpt_spatial_dim = kpt_spatial_dim
        self.focal_alpha = focal_alpha

        self.use_pose = opt_only is None or ("rot" in opt_only and "t" in opt_only)

        self.is_memotr = "memotr" in self.model_name
        if self.is_memotr:
            from memotr.models.criterion import build as build_criterion

            self.criterion = build_criterion(config=self.config, num_classes=num_classes)
            self.criterion.set_device(self.device)
        else:
            from trackformer.models.build import build_criterion
            from trackformer.models.matcher import build_matcher

            self.tf_args = get_trackformer_args(self.args)
            self.matcher = build_matcher(self.tf_args)
            self.criterion = build_criterion(
                self.tf_args,
                num_classes=num_classes + 1 if self.args.tf_use_focal_loss else num_classes + 1,
                matcher=self.matcher,
                device=self.args.device,
                use_rel_pose=self.args.do_predict_rel_pose,
            )

        self.init_optimizer()

        params_wo_grad = [
            (i, n) for i, (n, p) in enumerate(self.model_without_ddp.named_parameters()) if not p.requires_grad
        ]
        if len(params_wo_grad):
            self.logger.warning(f"Params without grad: {params_wo_grad}")
            self.logger.warning(f"{len(params_wo_grad)=}")

    def get_param_dicts(self):
        param_dicts = [
            {
                "params": [
                    p
                    for name, p in self.model_without_ddp.named_parameters()
                    if is_param_part_of_encoders(name, self.encoder_module_prefix)
                ],
                "lr": self.args.lr_encoders,
            },
            {
                "params": [
                    p
                    for name, p in self.model_without_ddp.named_parameters()
                    if not is_param_part_of_encoders(name, self.encoder_module_prefix)
                ],
                "lr": self.args.lr,
            },
        ]

        return param_dicts

    def freeze_encoder(self, model_without_ddp):
        for name, p in model_without_ddp.named_parameters():
            if is_param_part_of_encoders(name, self.encoder_module_prefix):
                p.requires_grad = False

    def loader_forward(
        self,
        *args,
        optimizer=None,
        stage="train",
        **kwargs,
    ):
        if stage == "train":
            optimizer = self.optimizer
        return super().loader_forward(
            *args,
            optimizer=optimizer,
            stage=stage,
            **kwargs,
        )

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

        seq_stats = defaultdict(list)
        seq_metrics = defaultdict(list)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq), disable=True)

        total_losses = []

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None  # processed ouput of the model that matches model's expected format
        pose_mat_prev_gt_abs = None
        pose_tokens_per_layer = None
        prev_tokens = None
        nan_count = 0
        do_skip_first_step = False
        prev_idx = None
        failed_ts = []

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["image"]
            targets = batch_t["target"]
            intrinsics = [x["intrinsics"] for x in targets]
            pts = batch_t["mesh_pts"]
            h, w = rgb.shape[-2:]

            if self.do_predict_rel_pose and t == 0:
                pose_prev_gt_abs = torch.cat([x["prev_target"]["pose"] for x in targets])
                pose_prev_pred_abs = {"t": pose_prev_gt_abs[..., :3], "rot": pose_prev_gt_abs[..., 3:]}

                if "detr" in self.model_name:
                    # trackformer concats t-1, t in forward, while std detr could use t-1,t tokens from the outer scope
                    model_forward_res = self.model_forward(
                        batch_t,
                        pose_tokens=pose_tokens_per_layer,
                        prev_tokens=prev_tokens,
                        use_prev_image=True,
                    )
                    prev_tokens = model_forward_res["out"]["tokens"]
            if self.use_pose_tokens and t == 0:
                model_forward_res = self.model_forward(batch_t, pose_tokens=pose_tokens_per_layer, use_prev_image=True)

                out = model_forward_res["out"]
                pose_tokens_per_layer = [o.unsqueeze(1) for o in out["pose_tokens"]]
                prev_tokens = out["tokens"]

            model_forward_res = self.model_forward(
                batch_t,
                pose_tokens=pose_tokens_per_layer,
                prev_tokens=prev_tokens,
            )
            out = model_forward_res["out"]

            # POSTPROCESS OUTPUTS

            if self.use_pose_tokens:
                pose_tokens_per_layer = (
                    [o.unsqueeze(0) for o in out["pose_tokens"]]
                    if pose_tokens_per_layer is None
                    else [
                        torch.cat([pose_tokens_per_layer[i], out["pose_tokens"][i].unsqueeze(1)], dim=1)[
                            :, -(max(1, self.seq_len - 1)) :
                        ]
                        for i in range(self.num_dec_layers)
                    ]
                )

            # LOSSES

            loss_dict = model_forward_res["loss_dict"]
            indices = loss_dict.pop("indices")
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            if self.use_kpt_loss:
                mask = batch_t["mask"]
                mask_dilated = mask.clone()
                if mask_dilated.shape[-1] == 3:
                    mask_dilated = mask_dilated.any(dim=-1)
                mask_dilated = torch.stack([mask_morph(m, kernel_size=5, op_name="dilate") for m in mask_dilated])
                score_map = out["score_map"].squeeze(1)
                mask_dilated = F.interpolate(
                    mask_dilated.unsqueeze(1).float(), size=score_map.shape[-2:], mode="nearest"
                ).squeeze(1)
                score_map[score_map < 0] = 0
                pos_weight = (mask_dilated == 0).float().sum() / (mask_dilated == 1).float().sum()
                loss_kpt = F.binary_cross_entropy(score_map, mask_dilated, weight=pos_weight)
                loss += 1 * loss_kpt
                seq_stats["loss_kpt"].append(loss_kpt.item())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
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

                optimizer.step()
            elif do_opt_in_the_end:
                total_losses.append(loss)

            # METRICS

            if self.use_pose:
                if self.do_predict_rel_pose:
                    indices = self.criterion.filter_out_idxs_for_rel_pose(targets, indices)
                idx = self.criterion._get_src_permutation_idx(indices)  # (batch_idx, query_idx)
                target_rts = torch.cat(
                    [torch.cat([t["t"][i], t["rot"][i]], dim=1) for t, (_, i) in zip(targets, indices)], dim=0
                )
                if len(target_rts) == 0:
                    pose_mat_gt_abs = torch.empty(0, 4, 4).to(self.device)
                    pose_mat_pred_abs = torch.empty(0, 4, 4).to(self.device)
                else:
                    other_values_for_metrics = self.get_req_target_values_for_metrics(targets, [i[1] for i in indices])

                    pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in target_rts])
                    t_pred = out["t"][idx]

                    if self.do_predict_2d_t:
                        center_depth_pred = out["center_depth"][idx]
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred, center_depth_pred, intrinsics, hw=(h, w)
                        )
                        t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

                    rot_pred = out["rot"][idx]
                    pred_rts = torch.cat([t_pred, rot_pred], dim=1)
                    pose_mat_pred = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pred_rts])

                    if self.do_predict_rel_pose:
                        pose_mat_prev_pred_abs = torch.stack(
                            [
                                self.pose_to_mat_converter_fn(rt)
                                for rt in torch.cat([pose_prev_pred_abs["t"], pose_prev_pred_abs["rot"]], dim=1)
                            ]
                        )
                        cur_bidx = idx[0]
                        if len(cur_bidx) > pose_mat_prev_pred_abs.shape[0]:
                            # reappears. use gt to fill in (TODO)
                            new_idx = [i for i in cur_bidx if i not in prev_idx[0]]
                            pose_mat_prev_pred_abs = torch.cat(
                                [pose_mat_prev_pred_abs, pose_mat_gt_abs[new_idx]], dim=0
                            )
                        elif len(cur_bidx) < pose_mat_prev_pred_abs.shape[0]:
                            # disappears
                            pose_mat_prev_pred_abs = pose_mat_prev_pred_abs[cur_bidx]
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

                    # since only pair seq for now
                    if self.do_predict_rel_pose:
                        # rel pose is undefined when obj is occluded at t-1 or t (const motion)
                        pose_mat_pred_metrics = pose_mat_pred
                        pose_mat_gt_rel = torch.cat(
                            [
                                self.pose_to_mat_converter_fn(t["pose_rel"][i])
                                for t, (_, i) in zip(targets, indices)
                                if len(t["pose_rel"]) > 0
                            ],
                            dim=0,
                        )
                        rot_mat_gt_rel = pose_mat_gt_rel[:, :3, :3]
                        t_gt_rel = pose_mat_gt_rel[:, :3, 3]
                        pose_mat_gt_metrics = convert_r_t_to_rt(rot_mat_gt_rel, t_gt_rel)
                    else:
                        pose_mat_pred_metrics = pose_mat_pred_abs
                        pose_mat_gt_metrics = pose_mat_gt_abs

                    m_batch_avg = self.calc_metrics_batch_mot(
                        pose_mat_pred_metrics=pose_mat_pred_metrics,
                        pose_mat_gt_metrics=pose_mat_gt_metrics,
                        **other_values_for_metrics,
                    )
                    for k, v in m_batch_avg.items():
                        if "classes" in k:
                            continue
                        seq_metrics[k].append(v)

                prev_idx = idx

            # UPDATE VARS

            if self.use_pose and seq_length > 1:
                if self.do_predict_rel_pose:
                    if self.do_predict_3d_rot:
                        rot_prev_pred_abs = matrix_to_axis_angle(rot_mat_pred_abs)
                    elif self.do_predict_6d_rot:
                        rot_prev_pred_abs = matrix_to_rotation_6d(rot_mat_pred_abs)
                    else:
                        rot_prev_pred_abs = matrix_to_quaternion(rot_mat_pred_abs)
                    pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
                else:
                    pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}
                if self.do_predict_2d_t:
                    pose_prev_pred_abs["center_depth"] = center_depth_pred
                pose_prev_pred_abs = {k: v.detach() for k, v in pose_prev_pred_abs.items()}

                pose_mat_prev_gt_abs = pose_mat_gt_abs

            # OTHER

            loss_value = losses_reduced_scaled.item()
            seq_stats["loss"].append(loss_value)
            for k, v in {**loss_dict_reduced_scaled}.items():
                if "indices" in k:
                    continue
                seq_stats[k].append(v.item())
            for k in [
                "class_error",
                "cardinality_error",
                "uncertainty",
                "confidence",
                "confidence_rot",
                "confidence_t",
            ]:
                if k in loss_dict_reduced:
                    v = loss_dict_reduced[k]
                    seq_stats[k].append(v.item())

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", np.mean(v), self.ts_counts_per_stage[stage])

            if not math.isfinite(loss_value):
                self.logger.error(f"Loss is {loss_value}, stopping training")
                self.logger.error(loss_dict_reduced)
                if t > 0:
                    self.logger.error(f"{batched_seq[t-1]=}")
                self.logger.error(f"{batched_seq[t]=}")
                if self.use_pose:
                    self.logger.error(f"rot_pred: {rot_pred}")
                self.logger.error(f"seq_metrics: {seq_metrics}")
                self.logger.error(f"seq_stats: {seq_stats}")
                sys.exit(1)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert self.use_pose
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                target_sizes = torch.stack([x["size"] for x in batch_t["target"]])
                out_formatted = postprocess_detr_outputs(
                    out, target_sizes=target_sizes, is_focal_loss=self.args.tf_use_focal_loss
                )
                # TODO: check n-obj case
                bboxs = []
                labels = []
                scores = []
                pose_preds = []
                for bidx, out_b in enumerate(out_formatted):
                    keep = out_b["scores"].cpu() > out_b["scores_no_object"].cpu()
                    # keep = torch.ones_like(res['scores']).bool()
                    if sum(keep) == 0:
                        failed_ts.append(t)
                        continue
                    boxes_b = out_b["boxes"][keep]
                    labels_b = out_b["labels"][keep]
                    scores_b = out_b["scores"][keep]
                    rot_pred_b = out_b["rot"][keep]
                    t_pred_b = out_b["t"][keep]

                    if self.do_predict_2d_t:
                        center_depth_pred = out_b["center_depth"][keep]
                        convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(
                            t_pred_b, center_depth_pred, intrinsics, hw=(h, w)
                        )
                        t_pred_b = convert_2d_t_pred_to_3d_res["t_pred"]
                    pose_pred_b = self.pose_to_mat_converter_fn(torch.cat([t_pred_b, rot_pred_b], dim=1))
                    pose_preds.append(pose_pred_b)
                    bboxs.append(boxes_b)
                    labels.append(labels_b)
                    scores.append(scores_b)
                assert batch_size == 1, batch_size
                if len(pose_preds) > 0:
                    pose_preds = torch.stack(pose_preds)[0]
                else:
                    pose_preds = torch.empty(0, 4, 4)
                det_res = {
                    "bbox": bboxs,
                    "labels": labels,
                    "scores": scores,
                }
                save_results_v2(
                    rgb,
                    intrinsics=intrinsics,
                    pose_gt=pose_mat_gt_abs,
                    pose_pred=pose_preds,
                    rgb_path=batch_t["rgb_path"],
                    preds_dir=preds_dir,
                    det_res=det_res,
                )

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["image", "mesh_bbox", "mask", "depth"]
                for k in vis_keys:
                    if len(batch_t.get(k, [])) == 0:
                        continue
                    vis_data[k].append(detach_and_cpu(batch_t[k]))
                vis_data["targets"].append(detach_and_cpu(targets))
                vis_data["out"].append(detach_and_cpu(out))
                if self.model_name == "detr_kpt":
                    vis_data["kpts"].append(detach_and_cpu(out["kpts"]))
                    vis_data["descriptors"].append(detach_and_cpu(out["descriptors"]))
                if self.use_pose:
                    vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs))

                # vis_data["pose_mat_pred_abs"].append(pose_mat_pred_abs[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_pred"].append(pose_mat_pred[vis_batch_idxs].detach().cpu())
                # vis_data["pose_mat_gt_abs"].append(pose_mat_gt_abs[vis_batch_idxs].cpu())

        if do_opt_in_the_end:
            total_loss = torch.mean(torch.stack(total_losses))
            total_loss.backward()
            unused_params = []
            if len(unused_params):
                self.logger.error(f"Unused params: {unused_params}")
                sys.exit(1)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if self.do_debug or do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                if do_vis:
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
            optimizer.step()
            optimizer.zero_grad()

        if not self.do_debug:
            for stats in [seq_stats, seq_metrics]:
                for k, v in stats.items():
                    stats[k] = np.mean(v)

        if self.use_pose and self.do_predict_rel_pose:
            # calc loss/metrics btw accumulated abs poses
            metrics_abs = self.calc_metrics_batch_mot(pose_mat_pred_abs, pose_mat_gt_abs, **other_values_for_metrics)
            for k, v in metrics_abs.items():
                seq_metrics[f"{k}_abs"].append(v)
            if not self.include_abs_pose_loss_for_rel:
                with torch.no_grad():

                    t_gt_abs = pose_mat_gt_abs[:, :3, 3]
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_gt_abs = self.rot_mat_to_vector_converter_fn(rot_mat_gt_abs)
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())

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

    def get_req_target_values_for_metrics(self, targets, indices):
        other_values_for_metrics_keys = ["mesh_pts", "mesh_bbox", "mesh_diameter", "is_sym"]
        other_values_for_metrics = {}
        for k in other_values_for_metrics_keys:
            if targets[0][k] is not None:
                v = torch.cat([t[k][i] for t, i in zip(targets, indices)], dim=0)
                if k == "is_sym":
                    v = ["full" if x else None for x in v]
                    k = "sym_types"
                other_values_for_metrics[k] = v

        return other_values_for_metrics

    def model_forward(self, batch_t, pose_tokens=None, prev_tokens=None, use_prev_image=False, **kwargs):
        def get_prev_data(key):
            if key not in batch_t["target"][0]:
                return []
            return [batch_t["target"][i][key] for i in range(len(batch_t["target"]))]

        if use_prev_image:
            targets = get_prev_data("prev_target")
            image = torch.stack(get_prev_data("prev_image"))
            depth = get_prev_data("prev_depth")
            if len(depth) > 0 and len(depth[0]) > 0:
                depth = torch.stack(depth)
            mask = get_prev_data("prev_mask")
            if len(mask) > 0 and len(mask[0]) > 0:
                mask = torch.stack(mask)
        else:
            targets = batch_t["target"]
            image = batch_t["image"]
            depth = batch_t["depth"]
            mask = batch_t["mask"]
        intrinsics = torch.stack([x["intrinsics"] for x in targets])

        extra_kwargs = {}
        if self.model_without_ddp.use_depth:
            extra_kwargs["depth"] = depth

        try:
            if self.model_name == "detr_kpt":
                if self.do_calibrate_kpt or self.kpt_spatial_dim > 2:
                    extra_kwargs["intrinsics"] = intrinsics.to(self.device)
                if self.kpt_spatial_dim > 2:
                    extra_kwargs["depth"] = depth
                out = self.model(
                    image,
                    mask=mask,
                    pose_tokens=pose_tokens,
                    prev_tokens=prev_tokens,
                    **extra_kwargs,
                )
            else:
                out = self.model(image, pose_tokens=pose_tokens, prev_tokens=prev_tokens, **extra_kwargs)

            if use_prev_image:
                loss_dict = {}
            else:
                loss_dict = self.criterion(out, targets)
        except Exception as e:
            self.logger.error(f"Loss calculation failed for batch {batch_t=}")
            self.logger.error(f"{targets=}")
            if "out" in locals():
                self.logger.error(f"{out=}")
            raise e

        return {"out": out, "loss_dict": loss_dict}

    def calc_metrics_batch_mot(
        self, pose_mat_pred_metrics, pose_mat_gt_metrics, mesh_pts, mesh_bbox, mesh_diameter, sym_types=None
    ):
        m_batch = defaultdict(list)
        for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_mat_pred_metrics, pose_mat_gt_metrics)):
            m_sample = calc_metrics(
                pred_rt=pred_rt,
                gt_rt=gt_rt,
                pts=mesh_pts[sample_idx],
                class_name=None,
                use_miou=True,
                bbox_3d=mesh_bbox[sample_idx],
                diameter=mesh_diameter[sample_idx],
                is_meters=True,
                use_m=self.use_m_for_metrics,
                sym_type=sym_types[sample_idx] if sym_types is not None else None,
                log_fn=print if self.logger is None else self.logger.warning,
            )
            for k, v in m_sample.items():
                m_batch[k].append(v)

        m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
        return m_batch_avg


class TrainerTrackformer(TrainerDeformableDETR):

    def __init__(
        self,
        *args_,
        **kwargs,
    ):

        super().__init__(*args_, **kwargs)

        param_dicts = self.get_param_dicts()

        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.args.detr_args.lr,
            weight_decay=self.args.weight_decay,
        )

    def get_param_dicts(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model_without_ddp.named_parameters()
                    if not match_module_by_name(
                        n,
                        self.args.detr_args.lr_backbone_names
                        + self.args.detr_args.lr_linear_proj_names
                        + ["layers_track_attention"],
                    )
                    and p.requires_grad
                ],
                "lr": self.args.detr_args.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model_without_ddp.named_parameters()
                    if match_module_by_name(n, self.args.detr_args.lr_backbone_names) and p.requires_grad
                ],
                "lr": self.args.detr_args.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.model_without_ddp.named_parameters()
                    if match_module_by_name(n, self.args.detr_args.lr_linear_proj_names) and p.requires_grad
                ],
                "lr": self.args.detr_args.lr * self.args.detr_args.lr_linear_proj_mult,
            },
        ]
        if self.args.detr_args.track_attention:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in self.model_without_ddp.named_parameters()
                        if match_module_by_name(n, ["layers_track_attention"]) and p.requires_grad
                    ],
                    "lr": self.args.detr_args.lr_track,
                }
            )

        return param_dicts

    def model_forward(self, batch_t, prev_features=None, **kwargs):
        rgb = batch_t["image"]
        targets = batch_t["target"]
        # when .eval(), clears all track_* in targets in detr_tracking
        out, targets_res, features, *_ = self.model(rgb, targets, prev_features=prev_features)
        loss_dict = self.criterion(out, targets_res)
        return {
            "out": out,
            "loss_dict": loss_dict,
            "targets_res": targets_res,
            "features": features,
        }
