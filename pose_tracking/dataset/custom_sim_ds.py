"""
.
├── intrinsics.txt
├── depth
│   └── *.png
├── instance_segmentation_fast
│   └── *.png
├── mesh
│   ├── *.obj
│   └── *.usd
└── rgb
    └── *.png
"""

import glob
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
from pose_tracking.dataset.ds_common import get_ds_sample
import torch
import trimesh
from pose_tracking.config import logger
from pose_tracking.utils.io import load_color, load_depth, load_pose
from pose_tracking.utils.trimesh_utils import load_mesh
from torch.utils.data import Dataset


class CustomSimDataset(Dataset):
    def __init__(self, root_dir, include_masks=False, zfar=np.inf, transforms=None, mesh_path=None, cam_pose_path=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.include_masks = include_masks
        self.color_files = sorted(glob.glob(f"{self.root_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{self.root_dir}/intrinsics.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = self.load_mask(self.color_files[0])
        self.zfar = zfar

        if cam_pose_path is None:
            if os.path.exists(f"{self.root_dir}/cam_pose.txt"):
                cam_pose_path = f"{self.root_dir}/cam_pose.txt"

        if cam_pose_path is not None:
            cam_pose_path = Path(cam_pose_path)
            self.cam_pose = load_pose(cam_pose_path)
        else:
            self.cam_pose = None

        self.mesh_path = mesh_path
        if mesh_path is not None:
            load_res = load_mesh(mesh_path)
            self.mesh = load_res["mesh"]
            self.mesh_bbox = load_res["bbox"]

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        path = self.color_files[idx]
        color = self.get_color(idx)
        depth_raw = self.get_depth(idx)
        
        if self.include_masks:
            bin_mask = self.load_mask(path)
        else:
            bin_mask = None

        sample = get_ds_sample(
            color, depth_raw, rgb_path=path, mask=bin_mask, intrinsics=self.K, transforms=self.transforms
        )

        return sample

    def load_mask(self, path):
        mask = load_color(path.replace("rgb/", "instance_segmentation_fast/"))
        obj_color = (255, 25, 25)
        mask = np.all(mask == obj_color, axis=-1).astype(bool)
        mask = torch.from_numpy(mask).float()
        # mask /= 255.0
        return mask

    def get_color(self, i):
        color = load_color(self.color_files[i], wh=(self.W, self.H))
        return color

    def get_depth(self, i):
        depth = load_depth(self.color_files[i].replace("rgb/", "depth/"), wh=(self.W, self.H), zfar=self.zfar)
        return depth


class CustomSimDatasetEval(CustomSimDataset):
    def __init__(self, preds_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds_path = Path(preds_path)
        self.pred_file_paths = list(sorted(self.preds_path.glob("*.txt")))
        if len(self.pred_file_paths) != len(self.color_files):
            logger.warning(
                f"Number of predictions ({len(self.pred_file_paths)}) does not match number of samples ({len(self.color_files)})"
            )

    def __len__(self):
        return min(len(self.color_files), len(self.pred_file_paths))

    def get_pred_pose(self, i):
        pose = load_pose(self.preds_path / f"{self.id_strs[i]}.txt")
        return pose

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample["pose_pred"] = self.get_pred_pose(i)
        return sample
