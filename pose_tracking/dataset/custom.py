"""
.
├── cam_K.txt
├── depth
│   └── *.png
├── masks
│   └── *.png
├── mesh
│   ├── *.obj
│   ├── *.obj.mtl
│   ├── images
│   │   └── *.pdf
│   └── textures
│       └── *.png
└── rgb
    └── *.png
"""

import glob
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh
from pose_tracking.config import logger
from pose_tracking.utils.io import load_color, load_depth, load_pose
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, zfar=np.inf, transforms=None, mesh_path=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.color_files = sorted(glob.glob(f"{self.root_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{self.root_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = cv2.imread(self.color_files[0].replace("rgb/", "masks/"), -1)
        self.zfar = zfar
        self.mesh_path = mesh_path
        if mesh_path is not None:
            self.mesh = self.load_mesh(mesh_path)

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        path = self.color_files[idx]
        color = self.get_color(idx)
        depth_raw = self.get_depth(idx)
        if self.transforms is None:
            rgb = torch.from_numpy(color)
        else:
            sample = self.transforms(image=color)
            rgb = sample["image"]

        rgb = rgb.float() / 255.0

        depth = torch.from_numpy(depth_raw).float()

        sample = {
            "rgb": rgb,
            "depth": depth,
            "rgb_path": path,
            "intrinsics": torch.from_numpy(self.K).float(),
        }

        return sample

    def get_color(self, i):
        color = load_color(self.color_files[i], wh=(self.W, self.H))
        return color

    def get_depth(self, i):
        depth = load_depth(self.color_files[i].replace("rgb/", "depth/"), wh=(self.W, self.H), zfar=self.zfar) / 1e3
        return depth

    def load_mesh(self, mesh_path, ext=None):
        if ext is None:
            ext = mesh_path.split(".")[-1]
        mesh = trimesh.load(open(mesh_path, "rb"), file_type=ext, force="mesh")
        bbox = mesh.bounding_box.vertices
        return {
            "mesh": mesh,
            "bbox": bbox,
        }


class CustomDatasetEval(CustomDataset):
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
