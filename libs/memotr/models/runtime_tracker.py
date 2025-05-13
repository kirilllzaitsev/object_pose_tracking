# Copyright (c) Ruopeng Gao. All Rights Reserved.
import copy
import os
from typing import Dict, List

from memotr.models.matcher import HungarianMatcher
import torch
from memotr.structures.track_instances import TrackInstances

from .motion import Motion
from .utils import logits_to_scores


class RuntimeTracker:
    def __init__(
        self,
        det_score_thresh: float = 0.7,
        track_score_thresh: float = 0.6,
        miss_tolerance: int = 5,
        use_motion: bool = False,
        motion_min_length: int = 3,
        motion_max_length: int = 5,
        visualize: bool = False,
        use_dab: bool = True,
        matcher:HungarianMatcher=None
    ):
        self.det_score_thresh = det_score_thresh
        self.track_score_thresh = track_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.use_motion = use_motion
        self.visualize = visualize
        self.motion_min_length = motion_min_length
        self.motion_max_length = motion_max_length
        self.motions: Dict[Motion] = {}
        self.use_dab = use_dab
        self.matcher=matcher

    def update(self, model_outputs: dict, tracks: List[TrackInstances], batch=None):
        assert len(tracks) == 1
        model_outputs["scores"] = logits_to_scores(model_outputs["pred_logits"])
        n_dets = len(model_outputs["det_query_embed"])

        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/runtime_tracker/", exist_ok=True)
            visualize_ids = tracks[0].ids.cpu().tolist()

        # Update tracks.
        tracks[0].boxes = model_outputs["pred_bboxes"][0][n_dets:]
        tracks[0].logits = model_outputs["pred_logits"][0][n_dets:]
        tracks[0].output_embed = model_outputs["outputs"][0][n_dets:]
        tracks[0].scores = logits_to_scores(tracks[0].logits)
        tracks[0].rots = model_outputs["rot"][0][n_dets:]
        tracks[0].ts = model_outputs["t"][0][n_dets:]
        for i in range(len(tracks[0])):
            if tracks[0].scores[i][tracks[0].labels[i]] < self.track_score_thresh:
                tracks[0].disappear_time[i] += 1
            else:
                if self.use_motion and tracks[0].disappear_time[i] > 0:
                    self.motions[tracks[0].ids[i].item()].clear()
                tracks[0].disappear_time[i] = 0
                if self.use_motion:
                    self.motions[tracks[0].ids[i].item()].add_box(
                        tracks[0].boxes[i].cpu()
                    )
                    tracks[0].last_appear_boxes[i] = tracks[0].boxes[i]
            if tracks[0].disappear_time[i] >= self.miss_tolerance:
                tracks[0].ids[i] = -1

        # Add newborn targets.
        new_tracks = TrackInstances(
            hidden_dim=tracks[0].hidden_dim, num_classes=tracks[0].num_classes
        )
        new_tracks_idxes = (
            torch.max(model_outputs["scores"][0][:n_dets], dim=-1).values
            >= self.det_score_thresh
        )
        new_tracks.logits = model_outputs["pred_logits"][0][:n_dets][new_tracks_idxes]
        new_tracks.boxes = model_outputs["pred_bboxes"][0][:n_dets][new_tracks_idxes]
        new_tracks.rots=model_outputs["rot"][0][:n_dets][new_tracks_idxes]
        new_tracks.ts=model_outputs["t"][0][:n_dets][new_tracks_idxes]
        new_tracks.ref_pts = model_outputs["last_ref_pts"][0][:n_dets][new_tracks_idxes]
        new_tracks.scores = model_outputs["scores"][0][:n_dets][new_tracks_idxes]
        new_tracks.output_embed = model_outputs["outputs"][0][:n_dets][new_tracks_idxes]
        # new_tracks.query_embed = model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
        if self.use_dab:
            new_tracks.query_embed = model_outputs["aux_outputs"][-1]["queries"][0][
                :n_dets
            ][new_tracks_idxes]
        else:
            new_tracks.query_embed = torch.cat(
                (
                    model_outputs["det_query_embed"][new_tracks_idxes][:, :256],  # hack
                    model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][
                        new_tracks_idxes
                    ],
                ),
                dim=-1,
            )
        new_tracks.disappear_time = torch.zeros(
            (len(new_tracks.logits),), dtype=torch.long
        )
        new_tracks.labels = torch.max(new_tracks.scores, dim=-1).indices

        if batch is not None and len(new_tracks) > 0 and self.matcher is not None:
            targets=batch['target']
            batch_size=1
            tracked_instances=copy.deepcopy(tracks)

            # 3. Get the detection results in current frame.
            detection_res = {
                "pred_logits": model_outputs["pred_logits"][:, :n_dets],
                "pred_boxes": model_outputs["pred_bboxes"][:, :n_dets],
                "rot": model_outputs["rot"][:, :n_dets],
                "t": model_outputs["t"][:, :n_dets],
            }

            gt_trackinstances = TrackInstances.init_tracks(
                batch,
                hidden_dim=new_tracks.hidden_dim,
                num_classes=new_tracks.num_classes,
            )
            for b in range(batch_size):
                gt_trackinstances[b].ids = batch["target"][b]["track_ids"]
                gt_trackinstances[b].labels = batch["target"][b]["labels"]
                gt_trackinstances[b].boxes = batch["target"][b]["boxes"]
                gt_trackinstances[b].rots = batch["target"][b]["rot"]
                gt_trackinstances[b].other_attrs["mesh_pts"] = batch["target"][b][
                    "mesh_pts"
                ]
                gt_trackinstances[b].other_attrs["mesh_bbox"] = batch["target"][b][
                    "mesh_bbox"
                ]
                gt_trackinstances[b].ts = batch["target"][b]["t"]

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
            # gt_full_idx = []
            # untracked_gt_trackinstances = []
            # for b in range(len(tracked_instances)):
            #     gt_full_idx.append(torch.arange(start=0, end=len(gt_trackinstances[b])))
            # for b in range(len(tracked_instances)):
            #     idx_bool = torch.ones(size=gt_full_idx[b].shape, dtype=torch.bool)
            #     for i in tracked_instances[b].matched_idx:
            #         if i.item() >= 0:
            #             idx_bool[i.item()] = False
            #     untracked_gt_trackinstances.append(gt_trackinstances[b][idx_bool])
            untracked_gt_trackinstances = gt_trackinstances

            # 5. Use Hungarian algorithm to matching.
            matcher_res = self.matcher(
                outputs=detection_res, targets=untracked_gt_trackinstances, use_focal=True
            )
            matcher_res = [list(mr) for mr in matcher_res]

            # match new_tracks and targets
            b=0
            new_tracks_idxes_numeric = torch.tensor([i for i, x in enumerate(new_tracks_idxes) if x])
            output_idx, gt_idx = matcher_res[b]
            matched_idx_mask = torch.isin(output_idx, new_tracks_idxes_numeric)
            # TODO (wrong matching wrt track queries)
            if len(gt_idx)>1:
                gt_idx=gt_idx[matched_idx_mask]
            # output_idx = output_idx[matched_idx_mask]
            gt_ids = untracked_gt_trackinstances[b].ids[gt_idx] # suppresses false pos
            gt_idx = torch.as_tensor(
                [gt_ids_to_idx[b][gt_id.item()] for gt_id in gt_ids], dtype=torch.long
            )
            # new_tracks.ids = gt_ids
            new_tracks.matched_idx = gt_idx
            new_tracks.other_attrs["mesh_pts"] = untracked_gt_trackinstances[
                b
            ].other_attrs["mesh_pts"][gt_idx]
            new_tracks.other_attrs["mesh_bbox"] = untracked_gt_trackinstances[
                b
            ].other_attrs["mesh_bbox"][gt_idx]
        # We do not use this post-precess motion module in our final version,
        # this will bring a slight improvement,
        # but makes un-elegant.
        if self.use_motion:
            new_tracks.last_appear_boxes = model_outputs["pred_bboxes"][0][:n_dets][
                new_tracks_idxes
            ]
        # V1
        ids = []
        for i in range(len(new_tracks)):
            ids.append(self.max_obj_id)
            self.max_obj_id += 1
        new_tracks.ids = torch.as_tensor(ids, dtype=torch.long)
        new_tracks = new_tracks.to(new_tracks.logits.device)
        for _ in range(len(new_tracks)):
            self.motions[new_tracks.ids[_].item()] = Motion(
                min_record_length=self.motion_min_length,
                max_record_length=self.motion_max_length,
            )
            self.motions[new_tracks.ids[_].item()].add_box(new_tracks.boxes[_].cpu())

        if self.visualize:
            visualize_ids += ids
            torch.save(
                torch.as_tensor(visualize_ids),
                "./outputs/visualize_tmp/runtime_tracker/ids.tensor",
            )

        return tracks, [new_tracks]
