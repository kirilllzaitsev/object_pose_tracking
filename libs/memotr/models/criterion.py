# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import os
from typing import Dict, List, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from pose_tracking.config import PROJ_DIR
from pose_tracking.dataset.ds_meta import YCBV_OBJ_ID_TO_NAME, YCBV_SYMMETRY_OBJ_NAMES
from pose_tracking.losses import compute_add_loss, compute_adds_loss, geodesic_loss_mat
from pose_tracking.metrics import accuracy
from pose_tracking.utils.pose import convert_rot_vector_to_matrix

from memotr.structures.track_instances import TrackInstances
from memotr.utils.box_ops import box_cxcywh_to_xyxy, box_iou_union, generalized_box_iou
from memotr.utils.utils import distributed_world_size, is_distributed

from .matcher import HungarianMatcher
from .matcher import build as build_matcher
from pose_tracking.utils.misc import print_cls

class ClipCriterion:
    def __init__(
        self,
        num_classes,
        matcher: HungarianMatcher,
        n_det_queries,
        aux_loss: bool,
        weight: dict,
        max_frame_length: int,
        n_aux: int,
        merge_det_track_layer: int = 0,
        aux_weights: List = None,
        hidden_dim: int = 256,
        use_dab: bool = True,
        rot_out_dim=4,
        t_out_dim=3,
        WITH_BOX_REFINE=True,
        use_rel_pose=False,
        rot_loss_name="mse",
        uncertainty_coef=0.1,
    ):
        """
        Init a criterion function.

        Args:
            num_classes: class num.
            matcher: matcher from DETR.
            n_det_queries: how many detection queries.
            aux_loss: whether use aux loss.
            weight: include "box_l1_loss", "box_giou_loss", "label_focal_loss"
        """
        self.device: None | torch.device = None
        self.aux_loss = aux_loss
        self.weight = weight
        self.num_classes = num_classes
        self.matcher = matcher
        self.n_det_queries = n_det_queries
        self.max_frame_length = max_frame_length
        self.n_aux = n_aux
        self.use_dab = use_dab
        self.frame_weights = [
            1.0
        ] * 1000  # if you want to set different weights for different frames
        self.aux_weights = aux_weights  # different weights for different DETR layers
        self.hidden_dim = hidden_dim
        self.merge_det_track_layer = merge_det_track_layer
        self.rot_out_dim = rot_out_dim
        self.t_out_dim = t_out_dim
        self.WITH_BOX_REFINE = WITH_BOX_REFINE
        self.uncertainty_coef = uncertainty_coef

        self.gt_trackinstances_list: None | List[List[TrackInstances]] = (
            None  # (clip_size, B)
        )
        self.loss = {}
        self.log = {}
        self.n_gts = []
        self.img_key = "image"

        self.use_rel_pose = use_rel_pose
        self.rot_loss_name = rot_loss_name

        # if rot_loss_name in ['adds']:
        #     self.pts = torch.load(f"{PROJ_DIR}/mesh_pts.pt")['dextreme']

    def set_device(self, device: torch.device):
        self.device = device

    def init_a_clip(
        self, batch: Dict, hidden_dim: int, num_classes: int, device: torch.device
    ):
        """
        Init this function for a specific clip.
        Args:
            batch: a batch data.
            hidden_dim:
            num_classes:
            device:
        Returns:
        """
        self.device = device
        clip_size = len(batch[self.img_key][0])
        batch_size = len(batch[self.img_key])
        self.gt_trackinstances_list = []
        for c in range(clip_size):
            gt_trackinstances = TrackInstances.init_tracks(
                batch,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                device=self.device,
                rot_out_dim=self.rot_out_dim,
                t_out_dim=self.t_out_dim,
                WITH_BOX_REFINE=self.WITH_BOX_REFINE,
            )
            for b in range(batch_size):
                gt_trackinstances[b].ids = batch["target"][b][c]["track_ids"]
                gt_trackinstances[b].labels = batch["target"][b][c]["labels"]
                gt_trackinstances[b].boxes = batch["target"][b][c]["boxes"]
                gt_trackinstances[b].rots = batch["target"][b][c]["rot"]
                gt_trackinstances[b].other_attrs["mesh_pts"] = batch["target"][b][c][
                    "mesh_pts"
                ]
                gt_trackinstances[b].ts = batch["target"][b][c]["t"]
                gt_trackinstances[b] = gt_trackinstances[b].to(self.device)
            self.gt_trackinstances_list.append(gt_trackinstances)

        self.n_gts = []
        if self.aux_loss:
            self.loss = {
                "box_l1_loss": torch.zeros(()).to(self.device),
                "box_giou_loss": torch.zeros(()).to(self.device),
                "label_focal_loss": torch.zeros(()).to(self.device),
                "rot_loss": torch.zeros(()).to(self.device),
                "t_loss": torch.zeros(()).to(self.device),
                "aux_box_l1_loss": torch.zeros(()).to(self.device),
                "aux_box_giou_loss": torch.zeros(()).to(self.device),
                "aux_label_focal_loss": torch.zeros(()).to(self.device),
                "aux_rot_loss": torch.zeros(()).to(self.device),
                "aux_t_loss": torch.zeros(()).to(self.device),
            }
        else:
            self.loss = {
                "box_l1_loss": torch.zeros(()).to(self.device),
                "box_giou_loss": torch.zeros(()).to(self.device),
                "label_focal_loss": torch.zeros(()).to(self.device),
                "rot_loss": torch.zeros(()).to(self.device),
                "t_loss": torch.zeros(()).to(self.device),
            }
        return

    def get_sum_loss_dict(self, loss_dict: dict):
        def get_weight(loss_name):
            if "box_l1_loss" in loss_name:
                return self.weight["box_l1_loss"]
            elif "box_giou_loss" in loss_name:
                return self.weight["box_giou_loss"]
            elif "label_focal_loss" in loss_name:
                return self.weight["label_focal_loss"]
            elif "rot_loss" in loss_name:
                return self.weight["rot_loss"]
            elif "t_loss" in loss_name:
                return self.weight["t_loss"]

        loss = sum([get_weight(k) * v for k, v in loss_dict.items()])
        return loss

    def get_mean_by_n_gts(self) -> Tuple[Dict, Dict]:
        total_n_gts = sum(self.n_gts)
        total_n_gts = torch.as_tensor(
            total_n_gts, dtype=torch.float, device=self.device
        )
        n_gts = torch.as_tensor(self.n_gts, dtype=torch.float, device=self.device)
        if is_distributed():
            torch.distributed.all_reduce(total_n_gts)
            torch.distributed.all_reduce(n_gts)
        total_n_gts = torch.clamp(total_n_gts / distributed_world_size(), min=1).item()
        n_gts = torch.clamp(n_gts / distributed_world_size(), min=1).tolist()
        loss = {}
        for k in self.loss:
            loss[k] = self.loss[k] / total_n_gts
        log = {}
        for k in self.log:
            for i in range(len(n_gts)):
                if f"frame{i}" in k:
                    log[k] = (self.log[k] / n_gts[i], 1)
                    break
        return loss, log

    def process_single_frame(
        self,
        model_outputs: dict,
        tracked_instances: List[TrackInstances],
        frame_idx: int,
    ):
        """
        Process this criterion for a single frame.

        I know this part is really complex and hard to understand (T.T),
        I will modify these in a possible extension version of this work in the future,
        but it works, doesn't it? :)
        Args:
            model_outputs: outputs from DETR.
            tracked_instances: already tracked instances.
            frame_idx: frame_idx t.
        """
        # 1. Get the GTs in current t frame.
        gt_trackinstances = self.gt_trackinstances_list[frame_idx]

        # 2. Update the already tracked instances.
        tracked_instances = self.update_tracked_instances(
            model_outputs=model_outputs, tracked_instances=tracked_instances
        )

        # 3. Get the detection results in current frame.
        detection_res = {
            "pred_logits": model_outputs["pred_logits"][
                :, : self.n_det_queries, :
            ].detach(),  # (B, Nd, n_classes)
            "pred_boxes": model_outputs["pred_bboxes"][
                :, : self.n_det_queries, :
            ].detach(),  # (B, Nd, 4)
            "rot": model_outputs["rot"][:, : self.n_det_queries, :].detach(),
            "t": model_outputs["t"][:, : self.n_det_queries, :].detach(),
        }

        # 4. Find some gts that do not include in the tracked instances mentioned in (2.),
        #    this gts need to be detected in current frame.
        gt_ids_to_idx = []
        for b in range(len(tracked_instances)):
            gt_ids_to_idx.append(
                {
                    gt_id.item(): gt_idx
                    for gt_idx, gt_id in enumerate(gt_trackinstances[b].ids)
                }
            )
        num_disappeared_tracked_gts = 0
        for b in range(len(tracked_instances)):
            gt_idx = []
            if len(tracked_instances[b]) > 0:
                for gt_id in tracked_instances[b].ids.tolist():
                    if gt_id in gt_ids_to_idx[b]:
                        gt_idx.append(gt_ids_to_idx[b][gt_id])
                    else:
                        gt_idx.append(-1)
                        num_disappeared_tracked_gts += 1
            tracked_instances[b].matched_idx = torch.as_tensor(
                data=gt_idx, dtype=tracked_instances[b].matched_idx.dtype
            )
        # 4.+ Filter the gts that not in the tracked instances:
        gt_full_idx = []
        untracked_gt_trackinstances = []
        for b in range(len(tracked_instances)):
            gt_full_idx.append(torch.arange(start=0, end=len(gt_trackinstances[b])))
        for b in range(len(tracked_instances)):
            idx_bool = torch.ones(size=gt_full_idx[b].shape, dtype=torch.bool)
            for i in tracked_instances[b].matched_idx:
                if i.item() >= 0:
                    idx_bool[i.item()] = False
            untracked_gt_trackinstances.append(gt_trackinstances[b][idx_bool])

        # 5. Use Hungarian algorithm to matching.
        matcher_res = self.matcher(
            outputs=detection_res, targets=untracked_gt_trackinstances, use_focal=True
        )
        matcher_res = [list(mr) for mr in matcher_res]

        def matcher_res_for_gt_idx(res):
            for bi in range(len(res)):
                ids = untracked_gt_trackinstances[bi].ids[res[bi][1]]
                idx = []
                for _ in ids:  # 遍历 ID
                    idx.append(gt_ids_to_idx[bi][_.item()])
                res[bi][1] = torch.as_tensor(idx, dtype=torch.long)
            return res

        # 6. Use the matched results to generate the tracked instances.
        new_trackinstances = []  # len is B
        for b in range(len(tracked_instances)):
            trackinstances = TrackInstances(
                frame_height=tracked_instances[b].frame_height,
                frame_width=tracked_instances[b].frame_width,
                hidden_dim=tracked_instances[b].hidden_dim,
                num_classes=self.num_classes,
                # use_dab=self.use_dab,
                rot_out_dim=self.rot_out_dim,
                t_out_dim=self.t_out_dim,
                WITH_BOX_REFINE=self.WITH_BOX_REFINE,
            )
            output_idx, gt_idx = matcher_res[b]
            gt_ids = untracked_gt_trackinstances[b].ids[gt_idx]
            gt_idx = torch.as_tensor(
                [gt_ids_to_idx[b][gt_id.item()] for gt_id in gt_ids], dtype=torch.long
            )
            trackinstances.ids = gt_ids
            trackinstances.matched_idx = gt_idx
            trackinstances.other_attrs["mesh_pts"] = untracked_gt_trackinstances[
                b
            ].other_attrs["mesh_pts"]
            # trackinstances.query_embed = model_outputs["aux_outputs"][-1]["queries"][b][output_idx]
            if self.use_dab:
                trackinstances.query_embed = model_outputs["aux_outputs"][-1][
                    "queries"
                ][b][output_idx]
            else:
                trackinstances.query_embed = torch.cat(
                    (
                        model_outputs["det_query_embed"][output_idx][
                            :, : self.hidden_dim
                        ],
                        model_outputs["aux_outputs"][-1]["queries"][b][output_idx],
                    ),
                    dim=-1,
                )
            trackinstances.ref_pts = model_outputs["last_ref_pts"][b][output_idx]
            trackinstances.output_embed = model_outputs["outputs"][b][output_idx]
            trackinstances.boxes = model_outputs["pred_bboxes"][b][output_idx]
            trackinstances.logits = model_outputs["pred_logits"][b][output_idx]
            trackinstances.rots = model_outputs["rot"][b][output_idx]
            trackinstances.ts = model_outputs["t"][b][output_idx]
            trackinstances.iou = torch.zeros((len(gt_idx),), dtype=torch.float)
            trackinstances = trackinstances.to(self.device)
            new_trackinstances.append(trackinstances)

        # 7. Add tracked instances to the matcher res, for loss computing.
        matcher_res = matcher_res_for_gt_idx(matcher_res)
        tracked_idx_to_gts_idx = []
        for b in range(len(tracked_instances)):
            tracked_outputs_idx = torch.arange(
                start=self.n_det_queries,
                end=self.n_det_queries + len(tracked_instances[b]),
            )
            tracked_gts_idx = tracked_instances[b].matched_idx
            tracked_idx_to_gts_idx.append([tracked_outputs_idx, tracked_gts_idx])
            assert len(tracked_outputs_idx) == len(tracked_gts_idx)
        outputs_idx_to_gts_idx = copy.deepcopy(matcher_res)
        for b in range(len(tracked_instances)):
            outputs_idx_to_gts_idx[b][0] = torch.cat(
                (outputs_idx_to_gts_idx[b][0], tracked_idx_to_gts_idx[b][0])
            )
            outputs_idx_to_gts_idx[b][1] = torch.cat(
                (outputs_idx_to_gts_idx[b][1], tracked_idx_to_gts_idx[b][1])
            )

        matched_idxs = [
            (idx_b[gt_idx_b >= 0], gt_idx_b[gt_idx_b >= 0])
            for (idx_b, gt_idx_b) in outputs_idx_to_gts_idx
        ]

        # 8. Compute the classification loss.
        loss_label = self.get_loss_label(
            outputs=model_outputs,
            gt_trackinstances=gt_trackinstances,
            idx_to_gts_idx=outputs_idx_to_gts_idx,
        )

        # 9. Compute the bounding box loss.
        loss_l1, loss_giou = self.get_loss_box(
            outputs=model_outputs,
            gt_trackinstances=gt_trackinstances,
            idx_to_gts_idx=outputs_idx_to_gts_idx,
        )

        loss_rot = self.get_loss_rot(
            outputs=model_outputs,
            gt_trackinstances=gt_trackinstances,
            idx_to_gts_idx=outputs_idx_to_gts_idx,
        )

        loss_t = self.get_loss_t(
            outputs=model_outputs,
            gt_trackinstances=gt_trackinstances,
            idx_to_gts_idx=outputs_idx_to_gts_idx,
        )

        # 10. Count how many GTs.
        n_gts = sum([len(gts) for gts in gt_trackinstances])
        self.loss["box_l1_loss"] += loss_l1 * self.frame_weights[frame_idx]
        self.loss["box_giou_loss"] += loss_giou * self.frame_weights[frame_idx]
        self.loss["label_focal_loss"] += loss_label * self.frame_weights[frame_idx]
        self.loss["rot_loss"] += loss_rot * self.frame_weights[frame_idx]
        self.loss["t_loss"] += loss_t * self.frame_weights[frame_idx]
        # Update logs.
        self.log[f"frame{frame_idx}_box_l1_loss"] = loss_l1.item()
        self.log[f"frame{frame_idx}_box_giou_loss"] = loss_giou.item()
        self.log[f"frame{frame_idx}_label_focal_loss"] = loss_label.item()
        self.log[f"frame{frame_idx}_rot_loss"] = loss_rot.item()
        self.log[f"frame{frame_idx}_t_loss"] = loss_t.item()
        self.n_gts.append(n_gts)

        # 11. Compute aux loss.
        if self.aux_loss:
            for i, aux_outputs in enumerate(model_outputs["aux_outputs"]):
                # Same to 3.
                aux_det_res = {
                    "pred_logits": aux_outputs["pred_logits"][
                        :, : self.n_det_queries, :
                    ].detach(),
                    "pred_boxes": aux_outputs["pred_bboxes"][
                        :, : self.n_det_queries, :
                    ].detach(),
                    "rot": aux_outputs["rot"][:, : self.n_det_queries, :].detach(),
                    "t": aux_outputs["t"][:, : self.n_det_queries, :].detach(),
                }
                # Same to 5.
                if i < self.merge_det_track_layer:
                    aux_matcher_res = self.matcher(
                        outputs=aux_det_res, targets=gt_trackinstances, use_focal=True
                    )
                    aux_matcher_res = [list(mr) for mr in aux_matcher_res]
                else:
                    aux_matcher_res = self.matcher(
                        outputs=aux_det_res,
                        targets=untracked_gt_trackinstances,
                        use_focal=True,
                    )
                    aux_matcher_res = [list(mr) for mr in aux_matcher_res]
                    aux_matcher_res = matcher_res_for_gt_idx(aux_matcher_res)
                # Same to some part in 7.
                aux_idx_to_gts_idx = copy.deepcopy(aux_matcher_res)
                for b in range(len(tracked_instances)):
                    if i < self.merge_det_track_layer:
                        aux_idx_to_gts_idx[b][0] = aux_idx_to_gts_idx[b][0]
                        aux_idx_to_gts_idx[b][1] = aux_idx_to_gts_idx[b][1]
                    else:
                        aux_idx_to_gts_idx[b][0] = torch.cat(
                            (aux_idx_to_gts_idx[b][0], tracked_idx_to_gts_idx[b][0])
                        )
                        aux_idx_to_gts_idx[b][1] = torch.cat(
                            (aux_idx_to_gts_idx[b][1], tracked_idx_to_gts_idx[b][1])
                        )

                # Compute the aux loss.
                aux_loss_label = self.get_loss_label(
                    outputs=model_outputs["aux_outputs"][i],
                    gt_trackinstances=gt_trackinstances,
                    idx_to_gts_idx=aux_idx_to_gts_idx,
                )
                aux_loss_l1, aux_loss_giou = self.get_loss_box(
                    outputs=model_outputs["aux_outputs"][i],
                    gt_trackinstances=gt_trackinstances,
                    idx_to_gts_idx=aux_idx_to_gts_idx,
                )
                aux_loss_rot = self.get_loss_rot(
                    outputs=model_outputs["aux_outputs"][i],
                    gt_trackinstances=gt_trackinstances,
                    idx_to_gts_idx=aux_idx_to_gts_idx,
                )
                aux_loss_t = self.get_loss_t(
                    outputs=model_outputs["aux_outputs"][i],
                    gt_trackinstances=gt_trackinstances,
                    idx_to_gts_idx=aux_idx_to_gts_idx,
                )

                self.loss["aux_box_l1_loss"] += (
                    aux_loss_l1 * self.frame_weights[frame_idx] * self.aux_weights[i]
                )
                self.loss["aux_box_giou_loss"] += (
                    aux_loss_giou * self.frame_weights[frame_idx] * self.aux_weights[i]
                )
                self.loss["aux_label_focal_loss"] += (
                    aux_loss_label * self.frame_weights[frame_idx] * self.aux_weights[i]
                )
                self.loss["aux_rot_loss"] += (
                    aux_loss_rot * self.frame_weights[frame_idx] * self.aux_weights[i]
                )
                self.loss["aux_t_loss"] += (
                    aux_loss_t * self.frame_weights[frame_idx] * self.aux_weights[i]
                )

        # Prepare the unmatched detection results.
        unmatched_detections = []
        for b in range(len(tracked_instances)):
            matched_indexes = set(outputs_idx_to_gts_idx[b][0].tolist())
            indexes = set([_ for _ in range(len(model_outputs["det_query_embed"]))])
            unmatched_indexes = list(indexes - matched_indexes)
            unmatched_indexes = torch.as_tensor(unmatched_indexes, dtype=torch.long)
            detections = TrackInstances(
                hidden_dim=model_outputs["outputs"].shape[-1],
                num_classes=model_outputs["pred_logits"].shape[-1],
                WITH_BOX_REFINE=self.WITH_BOX_REFINE,
                rot_out_dim=self.rot_out_dim,
                t_out_dim=self.t_out_dim,
            ).to(model_outputs["outputs"].device)
            detections.ref_pts = model_outputs["init_ref_pts"][b][unmatched_indexes]
            detections.output_embed = model_outputs["outputs"][b][unmatched_indexes]
            detections.logits = model_outputs["pred_logits"][b][unmatched_indexes]
            detections.boxes = model_outputs["pred_bboxes"][b][unmatched_indexes]
            detections.rots = model_outputs["rot"][b][unmatched_indexes]
            detections.ts = model_outputs["t"][b][unmatched_indexes]
            # detections.query_embed = model_outputs["aux_outputs"][-1]["queries"][b][unmatched_indexes]
            if self.use_dab:
                detections.query_embed = model_outputs["aux_outputs"][-1]["queries"][b][
                    unmatched_indexes
                ]
            else:
                detections.query_embed = torch.cat(
                    (
                        model_outputs["det_query_embed"][unmatched_indexes][
                            :, : self.hidden_dim
                        ],
                        model_outputs["aux_outputs"][-1]["queries"][b][
                            unmatched_indexes
                        ],
                    ),
                    dim=-1,
                )
            detections.ids = -torch.ones(
                (len(detections.query_embed),), dtype=torch.long, device=self.device
            )
            detections.matched_idx = -torch.ones(
                (len(detections.query_embed),), dtype=torch.long, device=self.device
            )
            detections.iou = torch.zeros(
                (
                    len(
                        detections.ids,
                    )
                ),
                dtype=torch.float,
                device=self.device,
            )
            unmatched_detections.append(detections)
            pass

        # Move to device.
        for b in range(len(tracked_instances)):
            tracked_instances[b] = tracked_instances[b].to(self.device)
            new_trackinstances[b] = new_trackinstances[b].to(self.device)

        # Compute IoU.
        for b in range(len(tracked_instances)):
            new_trackinstances[b].iou[new_trackinstances[b].matched_idx >= 0] = (
                torch.diag(
                    box_iou_union(
                        box_cxcywh_to_xyxy(
                            new_trackinstances[b][
                                new_trackinstances[b].matched_idx >= 0
                            ].boxes
                        ),
                        box_cxcywh_to_xyxy(
                            gt_trackinstances[b][
                                new_trackinstances[b][
                                    new_trackinstances[b].matched_idx >= 0
                                ].matched_idx
                            ].boxes
                        ),
                    )[0]
                )
            )
            tracked_instances[b].iou[tracked_instances[b].matched_idx >= 0] = (
                torch.diag(
                    box_iou_union(
                        box_cxcywh_to_xyxy(
                            tracked_instances[b][
                                tracked_instances[b].matched_idx >= 0
                            ].boxes
                        ),
                        box_cxcywh_to_xyxy(
                            gt_trackinstances[b][
                                tracked_instances[b][
                                    tracked_instances[b].matched_idx >= 0
                                ].matched_idx
                            ].boxes
                        ),
                    )[0]
                )
            )
            pass

        # return tracked_instances, new_trackinstances, unmatched_detections
        return {
            "tracked_instances": tracked_instances,
            "new_trackinstances": new_trackinstances,
            "unmatched_detections": unmatched_detections,
            "matched_idxs": matched_idxs,
        }

    def update_tracked_instances(
        self, model_outputs: dict, tracked_instances: List[TrackInstances]
    ) -> List[TrackInstances]:
        """
        Update tracked instances.
        """
        for b in range(len(tracked_instances)):
            if len(tracked_instances[b]) > 0:
                track_mask = model_outputs["query_mask"][b][self.n_det_queries :]
                tracked_instances[b].boxes = model_outputs["pred_bboxes"][b][
                    self.n_det_queries :
                ][~track_mask]
                tracked_instances[b].logits = model_outputs["pred_logits"][b][
                    self.n_det_queries :
                ][~track_mask]
                tracked_instances[b].rots = model_outputs["rot"][b][
                    self.n_det_queries :
                ][~track_mask]
                tracked_instances[b].ts = model_outputs["t"][b][self.n_det_queries :][
                    ~track_mask
                ]
                # Query embed and ref_pts will be updated in the query_updater module.
                tracked_instances[b].output_embed = model_outputs["outputs"][b][
                    self.n_det_queries :
                ][~track_mask]
                tracked_instances[b].matched_idx = torch.zeros(
                    (0,), dtype=tracked_instances[b].matched_idx.dtype
                )
                tracked_instances[b].labels = torch.zeros(
                    (0,), dtype=tracked_instances[b].matched_idx.dtype
                )
        return tracked_instances

    def get_loss_label(
        self, outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx
    ):
        """
        Compute the classification loss.
        """
        pred_logits = [
            preds[~mask]
            for preds, mask in zip(outputs["pred_logits"], outputs["query_mask"])
        ]
        gt_labels = [
            torch.full(
                (pred_logits[b].shape[:1]),
                self.num_classes,
                dtype=torch.int64,
                device=self.device,
            )
            for b in range(len(gt_trackinstances))
        ]
        for b in range(len(pred_logits)):
            gt_labels[b][idx_to_gts_idx[b][0][idx_to_gts_idx[b][1] >= 0]] = (
                gt_trackinstances[b].labels[
                    idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]
                ]
            )
        pred_logits = torch.cat(pred_logits)
        gt_labels = torch.cat(gt_labels)
        gt_labels_one_hot = (
            F.one_hot(gt_labels, self.num_classes + 1)[:, :-1]
            .to(pred_logits.dtype)
            .to(pred_logits.device)
        )

        loss = sigmoid_focal_loss(
            inputs=pred_logits, targets=gt_labels_one_hot, alpha=0.25, gamma=2
        )

        return loss

    @staticmethod
    def get_loss_box(outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx):
        """
        Computer the bounding box loss, l1 and giou.
        """
        matched_pred_boxes = [
            boxes[outputs_idx[0][outputs_idx[1] >= 0]]
            for boxes, outputs_idx in zip(outputs["pred_bboxes"], idx_to_gts_idx)
        ]
        gt_boxes = [
            gt_trackinstances[b].boxes[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        matched_pred_boxes = torch.cat(matched_pred_boxes)
        gt_boxes = torch.cat(gt_boxes).to(matched_pred_boxes.device)

        loss_l1 = F.mse_loss(
            input=matched_pred_boxes, target=gt_boxes, reduction="none"
        ).sum()
        # TODO: check is gt cxcy?
        loss_giou = (
            1
            - torch.diag(
                input=generalized_box_iou(
                    box_cxcywh_to_xyxy(matched_pred_boxes), box_cxcywh_to_xyxy(gt_boxes)
                )
            )
        ).sum()
        return loss_l1, loss_giou

    # @staticmethod
    def get_loss_rot(
        self, outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx
    ):
        """
        Computer the bounding rot loss
        """
        matched_pred_rots = [
            rots[outputs_idx[0][outputs_idx[1] >= 0]]
            for rots, outputs_idx in zip(outputs["rot"], idx_to_gts_idx)
        ]
        gt_rots = [
            gt_trackinstances[b].rots[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]
        matched_pred_rots = torch.cat(matched_pred_rots)
        gt_rots = torch.cat(gt_rots).to(matched_pred_rots.device)

        if self.rot_loss_name in ["adds", "geodesic_mat"]:

            is_sym = torch.cat(
                [
                    torch.tensor([YCBV_OBJ_ID_TO_NAME[oid] in YCBV_SYMMETRY_OBJ_NAMES])
                    for oid in torch.cat(
                        [
                            gt_trackinstances[b].labels[
                                idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]
                            ]
                            for b in range(len(gt_trackinstances))
                        ]
                    ).tolist()
                ]
            )
            if self.rot_loss_name == "adds":
                mesh_pts = torch.cat(
                    [
                        gt_trackinstances[b].other_attrs["mesh_pts"][
                            idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]
                        ]
                        for b in range(len(gt_trackinstances))
                    ]
                )
                mesh_pts_sym = mesh_pts[is_sym]
                mesh_pts_asym = mesh_pts[~is_sym]
            matched_pred_rot_mats = convert_rot_vector_to_matrix(matched_pred_rots)
            gt_rot_mats = convert_rot_vector_to_matrix(gt_rots)
            pred_rot_mats_sym = matched_pred_rot_mats[is_sym]
            gt_rot_mats_sym = gt_rot_mats[is_sym]
            pred_rot_mats_asym = matched_pred_rot_mats[~is_sym]
            gt_rot_mats_asym = gt_rot_mats[~is_sym]
            loss_l1 = torch.tensor(0.0).to(matched_pred_rot_mats.device)
            if (~is_sym).any():
                if self.rot_loss_name == "adds":
                    loss_l1_asym = compute_add_loss(
                        pred_rot_mats_asym,
                        pose_gt=gt_rot_mats_asym,
                        pts=mesh_pts_asym.to(matched_pred_rot_mats.device) * 1e1,
                    )
                else:
                    loss_l1_asym = 1e-1 * geodesic_loss_mat(
                        pred_rot_mats_asym, gt_rot_mats_asym
                    )
                loss_l1 += loss_l1_asym * (~is_sym).sum()  # later div by num gts
            if (is_sym).any():
                if self.rot_loss_name == "adds":
                    loss_l1_sym = compute_adds_loss(
                        pred_rot_mats_sym,
                        pose_gt=gt_rot_mats_sym,
                        pts=mesh_pts_sym.to(matched_pred_rot_mats.device) * 1e1,
                    )
                else:
                    loss_l1_sym = 1e-1 * geodesic_loss_mat(
                        pred_rot_mats_sym, gt_rot_mats_sym, sym_type="full"
                    )
                loss_l1 += loss_l1_sym * (is_sym).sum()
            if torch.isnan(loss_l1):
                print(f"{loss_l1=}")
                print(f"{matched_pred_rot_mats=}")
                print(f"{gt_rot_mats=}")
                raise RuntimeError("loss rot is nan")
        else:
            loss_l1 = F.mse_loss(
                input=matched_pred_rots, target=gt_rots, reduction="none"
            ).sum()

        return loss_l1

    # @staticmethod
    def get_loss_t(
        self, outputs, gt_trackinstances: List[TrackInstances], idx_to_gts_idx
    ):
        """
        Computer the bounding t loss
        """
        matched_pred_ts = [
            ts[outputs_idx[0][outputs_idx[1] >= 0]]
            for ts, outputs_idx in zip(outputs["t"], idx_to_gts_idx)
        ]
        gt_ts = [
            gt_trackinstances[b].ts[idx_to_gts_idx[b][1][idx_to_gts_idx[b][1] >= 0]]
            for b in range(len(gt_trackinstances))
        ]

        matched_pred_ts = torch.cat(matched_pred_ts)
        gt_ts = torch.cat(gt_ts).to(matched_pred_ts.device)

        if self.t_out_dim == 2:
            src_depths = torch.cat(
                [
                    ts[outputs_idx[0][outputs_idx[1] >= 0]]
                    for ts, outputs_idx in zip(outputs["center_depth"], idx_to_gts_idx)
                ]
            ).squeeze(-1)
            target_depths = gt_ts[..., -1]
            if len(target_depths) == 0:
                return torch.tensor(0).to(matched_pred_ts.device)
            # cxcy loss comes from bbox
            loss_l1 = F.mse_loss(src_depths, target_depths, reduction="none")
            loss_l1 = loss_l1.sum()
        else:
            loss_l1 = F.mse_loss(
                input=matched_pred_ts, target=gt_ts, reduction="none"
            ).sum()

        return loss_l1
    
    def __repr__(self):
        return print_cls(
            self,
            extra_str=super().__repr__(),
            excluded_attrs=[
                "query_embed",
                "long_memory",
                "last_output",
                "output_embed",
            ],
        )



def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()  # 在类别上计算平均


def build(config: dict, num_classes=None):
    dataset_num_classes = {
        "DanceTrack": 1,
        "SportsMOT": 1,
        "MOT17": 1,
        "MOT17_SPLIT": 1,
        "BDD100K": 8,
    }
    opt_only = config.get("opt_only", None)
    use_boxes = opt_only is not None and all(x in opt_only for x in ["boxes"])
    if not use_boxes:
        config["MATCH_COST_BBOX"] = 0.0
        config["MATCH_COST_GIOU"] = 0.0
        config["LOSS_WEIGHT_L1"] = 0.0
        config["LOSS_WEIGHT_GIOU"] = 0.0
    return ClipCriterion(
        num_classes=num_classes or dataset_num_classes[config["DATASET"]],
        matcher=build_matcher(config=config),
        n_det_queries=config["NUM_DET_QUERIES"],
        aux_loss=config["AUX_LOSS"],
        weight={
            "box_l1_loss": config["LOSS_WEIGHT_L1"],
            "box_giou_loss": config["LOSS_WEIGHT_GIOU"],
            "label_focal_loss": config["LOSS_WEIGHT_FOCAL"],
            # "box_l1_loss": 0,
            # "box_giou_loss": 0,
            # "label_focal_loss": 0,
            "rot_loss": config["LOSS_WEIGHT_ROT"],
            "t_loss": config["LOSS_WEIGHT_T"],
        },
        max_frame_length=max(config["SAMPLE_LENGTHS"]),
        n_aux=config["NUM_DEC_LAYERS"] - 1,
        merge_det_track_layer=(
            0
            if "MERGE_DET_TRACK_LAYER" not in config
            else config["MERGE_DET_TRACK_LAYER"]
        ),
        aux_weights=config["AUX_LOSS_WEIGHT"],
        hidden_dim=config["HIDDEN_DIM"],
        use_dab=config["USE_DAB"],
        rot_out_dim=config["rot_out_dim"],
        t_out_dim=config["t_out_dim"],
        WITH_BOX_REFINE=config["WITH_BOX_REFINE"],
        rot_loss_name=config["rot_loss_name"],
    )
