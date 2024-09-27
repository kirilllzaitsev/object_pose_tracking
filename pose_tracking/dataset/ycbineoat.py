import glob
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
from pose_tracking.config import logger
from pose_tracking.dataset.ds_meta import YCBINEOAT_VIDEONAME_TO_OBJ
from pose_tracking.utils.geom import backproj_depth
from torch.utils.data import Dataset


class YCBineoatDataset(Dataset):
    # https://github.com/NVlabs/FoundationPose/blob/main/datareader.py#L57
    def __init__(
        self,
        video_dir,
        downscale=1,
        shorter_side=None,
        zfar=np.inf,
        ycb_meshes_dir=None,
        include_rgb=True,
        include_depth=True,
        include_mask=True,
        include_xyz_map=False,
        include_occ_mask=False,
        include_gt_pose=True,
    ):
        self.video_dir = video_dir
        self.downscale = downscale
        self.ycb_meshes_dir = ycb_meshes_dir
        self.include_rgb = include_rgb
        self.include_mask = include_mask
        self.include_depth = include_depth
        self.include_xyz_map = include_xyz_map
        self.include_occ_mask = include_occ_mask
        self.include_gt_pose = include_gt_pose
        self.zfar = zfar
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{video_dir}/cam_K.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.h, self.w = cv2.imread(self.color_files[0]).shape[:2]

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.h, self.w)

        self.h = int(self.h * self.downscale)
        self.w = int(self.w * self.downscale)
        self.K[:2] *= self.downscale

        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, i):
        sample = {}

        if self.include_rgb:
            sample["color"] = self.get_color(i)
        if self.include_mask:
            sample["mask"] = self.get_mask(i)
        if self.include_depth:
            sample["depth"] = self.get_depth(i)
        if self.include_xyz_map:
            sample["xyz_map"] = self.get_xyz_map(i)
        if self.include_occ_mask:
            sample["occ_mask"] = self.get_occ_mask(i)
        if self.include_gt_pose:
            sample["pose"] = self.get_gt_pose(i)
        sample["intrinsics"] = self.K

        return sample

    def get_gt_pose(self, i):
        pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
        return pose

    def get_color(self, i):
        return get_color(self.color_files[i], wh=(self.w, self.h))

    def get_mask(self, i):
        return get_mask(self.color_files[i].replace("rgb", "masks"), wh=(self.w, self.h))

    def get_depth(self, i):
        return get_depth(self.color_files[i].replace("rgb", "depth"), wh=(self.w, self.h), zfar=self.zfar)

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = backproj_depth(depth, self.K, do_flip_xy=True)
        return xyz_map

    def get_occ_mask(self, i):
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.h, self.w), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(occ_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        assert self.ycb_meshes_dir is not None, "ycb_meshes_dir is not set"
        ob_name = YCBINEOAT_VIDEONAME_TO_OBJ[self.get_video_name()]
        mesh = trimesh.load(f"{self.ycb_meshes_dir}/{ob_name}/textured_simple.obj")
        return mesh

    def get_video_name(self):
        return self.video_dir.split("/")[-1]


class YCBineoatDatasetBenchmark(YCBineoatDataset):
    def __init__(self, preds_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds_path = Path(preds_path)

    def get_pred_pose(self, i):
        pose = np.loadtxt(self.preds_path / f"{self.id_strs[i]}.txt").reshape(4, 4)
        return pose

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample["pose_pred"] = self.get_pred_pose(i)
        return sample


def get_depth(path, wh=None, zfar=np.inf):
    depth = cv2.imread(path, -1) / 1e3
    if wh is not None:
        depth = resize_img(depth, wh=wh)
    depth[(depth < 0.001) | (depth >= zfar)] = 0
    return depth


def resize_img(depth, wh):
    return cv2.resize(depth, (wh[0], wh[1]), interpolation=cv2.INTER_NEAREST)


def get_color(path, wh=None):
    color = imageio.imread(path)[..., :3]
    if wh is not None:
        color = resize_img(color, wh=wh)
    return color


def get_mask(path, wh=None):
    mask = cv2.imread(path, -1)
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break
    if wh is not None:
        mask = resize_img(mask, wh=wh)
    mask = mask.astype(bool).astype(np.uint8)
    return mask
