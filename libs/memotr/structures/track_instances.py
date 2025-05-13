# Copyright (c) Ruopeng Gao. All Rights Reserved.
from collections import defaultdict
from typing import List

import torch
from pose_tracking.utils.misc import print_cls


class TrackInstances:
    """
    Tracked Instances.
    """

    rot_out_dim = 4
    t_out_dim = 3

    def __init__(
        self,
        frame_height: float = 1.0,
        frame_width: float = 1.0,
        hidden_dim: int = 256,
        num_classes: int = 1,
        use_dab: bool = False,
        WITH_BOX_REFINE=True,
        **kwargs
    ):
        self.use_dab = use_dab
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        if self.use_dab:
            self.ref_pts = torch.zeros((0, 4))
            self.query_embed = torch.zeros((0, hidden_dim))
        else:
            # self.ref_pts = torch.zeros((0, 2))
            self.ref_pts = (
                torch.zeros((0, 4)) if WITH_BOX_REFINE else torch.zeros((0, 2))
            )
            self.query_embed = torch.zeros((0, 2 * hidden_dim))
        self.ids = torch.zeros((0,), dtype=torch.long)
        self.boxes = torch.zeros((0, 4))
        self.labels = torch.zeros((0,), dtype=torch.long)
        self.logits = torch.zeros((0, self.num_classes))
        self.matched_idx = torch.zeros(
            (0,), dtype=torch.long
        )  # gt idxs that were matched to preds. -1 = non-matched
        self.output_embed = torch.zeros((0, self.hidden_dim))
        self.disappear_time = torch.zeros((0,), dtype=torch.long)
        self.scores = torch.zeros((0,), dtype=torch.float)
        self.area = torch.zeros((0,), dtype=torch.float)
        self.iou = torch.zeros((0,), dtype=torch.float)
        self.last_output = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.long_memory = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.last_appear_boxes = torch.zeros((0, 4))

        self.rots = torch.zeros((0, self.rot_out_dim))
        self.ts = torch.zeros((0, self.t_out_dim))
        self.other_attrs = {
            "mesh_pts": torch.zeros((0, 1000, 3)),
            "mesh_bbox": torch.zeros((0, 8, 3)),
        }

    def to(self, device):
        res = TrackInstances(
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        for k, v in vars(self).items():
            if hasattr(v, "to"):
                v = v.to(device)
            res.__setattr__(k, v)
        return res

    def __len__(self):
        assert self.ref_pts.shape[0] == self.query_embed.shape[0]
        return max(self.query_embed.shape[0], self.labels.shape[0])

    def __getitem__(self, item: int | slice | torch.BoolTensor) -> "TrackInstances":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("TrackInstances index out of range!")
            else:
                item = slice(item, None, len(self))
        res = TrackInstances(
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        for k, v in vars(self).items():
            if hasattr(v, "__getitem__") and hasattr(v, "shape") and v.shape[0] != 0:
                res.__setattr__(k, v[item])
            else:
                res.__setattr__(k, v)
        return res

    @staticmethod
    def init_tracks(
        batch: dict,
        hidden_dim: int,
        num_classes: int,
        device="cpu",
        use_dab: bool = False,
        rot_out_dim=4,
        t_out_dim=3,
        WITH_BOX_REFINE=True,
    ):
        """
        Init tracks for a batch.
        """
        tracks_list = []
        h_max, w_max = 0, 0
        for i in range(len(batch["image"])):
            h_max = max(batch["image"][i][0].shape[-2], h_max)
            w_max = max(batch["image"][i][0].shape[-1], w_max)
        for i in range(len(batch["image"])):
            tracks_list.append(
                TrackInstances(
                    frame_height=float(batch["image"][i][0].shape[-2] / h_max),
                    frame_width=float(batch["image"][i][0].shape[-1] / w_max),
                    hidden_dim=hidden_dim,
                    num_classes=num_classes,
                    use_dab=use_dab,
                    rot_out_dim=rot_out_dim,
                    t_out_dim=t_out_dim,
                    WITH_BOX_REFINE=WITH_BOX_REFINE,
                ).to(device)
            )
        return tracks_list

    @staticmethod
    def cat_tracked_instances(tracked1: "TrackInstances", tracked2: "TrackInstances"):
        res = TrackInstances(
            frame_height=tracked1.frame_height, frame_width=tracked1.frame_width
        )

        for k, v in vars(tracked2).items():
            if type(v) is torch.Tensor:
                a1 = getattr(tracked1, k)
                a2 = getattr(tracked2, k)
                if a1.device.type == "cuda" and a2.device.type == "cpu":
                    a2 = a2.to(a1.device)
                elif a1.device.type == "cpu" and a2.device.type == "cuda":
                    a1 = a1.to(a2.device)
                res.__setattr__(
                    k, torch.cat((a1, getattr(tracked2, k)))
                )
            elif type(v) in [dict, defaultdict]:
                res.__setattr__(k, defaultdict(list))
                new_k={}
                for k2, v2 in v.items():
                    if type(v2) is torch.Tensor:
                        a1 = getattr(tracked1, k)[k2]
                        a2 = getattr(tracked2, k)[k2]
                        if a1.device.type == "cuda" and a2.device.type == "cpu":
                            a2 = a2.to(a1.device)
                        elif a1.device.type == "cpu" and a2.device.type == "cuda":
                            a1 = a1.to(a2.device)
                        new_k[k2] = torch.cat((a1, a2))
                    else:
                        new_k[k2] = v2
                res.__setattr__(k, new_k)
        return res

    @staticmethod
    def tracks_to_meta_tensors(tracks: List["TrackInstances"]):
        keys = [
            k for k in vars(tracks[0]) if type(getattr(tracks[0], k)) is torch.Tensor
        ]
        meta = {
            "frame_height": [],
            "frame_width": [],
            "hidden_dim": [],
            "num_classes": [],
            "keys": keys,
        }
        tensors = []
        for b in range(len(tracks)):
            meta["frame_height"].append(tracks[b].frame_height)
            meta["frame_width"].append(tracks[b].frame_width)
            meta["hidden_dim"].append(tracks[b].hidden_dim)
            meta["num_classes"].append(tracks[b].num_classes)
            for k in keys:
                tensors.append(getattr(tracks[b], k))
        return meta, tensors

    @staticmethod
    def meta_tensors_to_tracks(meta: dict, tensors: List[torch.Tensor]):
        tracks = []
        for b in range(len(meta["frame_height"])):
            track = TrackInstances(
                frame_height=meta["frame_height"][b],
                frame_width=meta["frame_width"][b],
                hidden_dim=meta["hidden_dim"][b],
                num_classes=meta["num_classes"][b],
            )
            for i, k in enumerate(meta["keys"]):
                setattr(track, k, tensors[i + b * len(meta["keys"])])
            tracks.append(track)
        return tracks

    def clone(self):
        """
        Clone the TrackInstances.
        """
        res = TrackInstances(
            frame_height=self.frame_height,
            frame_width=self.frame_width,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )
        for k, v in vars(self).items():
            if hasattr(v, "clone"):
                res.__setattr__(k, v.clone())
            else:
                res.__setattr__(k, v)
        return res

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
