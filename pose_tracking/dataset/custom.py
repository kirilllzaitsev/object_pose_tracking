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


class CustomDataset(Dataset):
    def __init__(self, root_dir, zfar=np.inf):
        self.root_dir = root_dir
        self.color_files = sorted(glob.glob(f"{self.root_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{self.root_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = cv2.imread(self.color_files[0].replace("rgb/", "masks/"), -1)
        self.zfar = zfar

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color = self.get_color(idx)
        depth = self.get_depth(idx)
        sample = {
            "rgb": torch.from_numpy(color).permute(2, 0, 1).float(),
            "depth": torch.from_numpy(depth).float(),
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
