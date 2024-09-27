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

import cv2
import imageio
import numpy as np
import torch
from pose_tracking.config import logger
from torch.utils.data import Dataset
import trimesh


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
            "intrinsic": torch.from_numpy(self.K).float(),
        }

        return sample

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])[..., :3]
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_depth(self, i):
        depth = cv2.imread(self.color_files[i].replace("rgb/", "depth/"), -1) / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def load_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path, force="mesh")
        return mesh
