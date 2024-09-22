import glob
import os
import re

import cv2
import imageio
import numpy as np
import torch
from pose_tracking.config import logger
from torch.utils.data import Dataset


class BCOTDataset(Dataset):
    def __init__(self, root_dir, obj_name):
        self.root_dir = root_dir
        self.color_files = sorted(glob.glob(f"{self.root_dir}/{obj_name}/*.png"))
        self.K = self.parse_intrinsics(f"{self.root_dir}/K.txt")
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)

        self.poses = np.loadtxt(f"{self.root_dir}/{obj_name}/pose.txt")

        self.h, self.w = cv2.imread(self.color_files[0]).shape[:2]

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color = self.get_color(idx)
        pose_row = self.poses[idx]
        r, t = pose_row[:9].reshape(3, 3), pose_row[9:]
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = t

        sample = {
            "rgb": torch.from_numpy(color).permute(2, 0, 1).float(),
            "pose": torch.from_numpy(pose).float(),
        }

        return sample

    def get_color(self, i):
        color = imageio.imread(self.color_files[i])[..., :3]
        return color

    def parse_intrinsics(self, path):
        # Matx33f K = Matx33f(481.34025, 0.0, 329.4003, 0.0, 481.8243, 260.3788, 0.000000, 0.000000, 1.000000);
        p = re.compile(r"Matx33f\((.*), (.*), (.*), (.*), (.*), (.*), (.*), (.*), (.*)\);")
        content = open(path).read()
        m = p.search(content)
        if m is None:
            raise ValueError(f"Failed to parse intrinsics from {path}")
        return np.array([float(m.group(i)) for i in range(1, 10)]).reshape(3, 3)
