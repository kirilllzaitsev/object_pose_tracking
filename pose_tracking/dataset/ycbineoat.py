import copy
import glob
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import trimesh
from bop_toolkit_lib.inout import load_depth
from pose_tracking.config import logger
from pose_tracking.dataset.ds_common import process_raw_sample
from pose_tracking.dataset.ds_meta import YCBINEOAT_VIDEONAME_TO_OBJ
from pose_tracking.utils.geom import backproj_depth
from pose_tracking.utils.io import load_color, load_depth, load_mask, load_pose
from pose_tracking.utils.trimesh_utils import load_mesh
from torch.utils.data import Dataset


class YCBineoatDataset(Dataset):

    ds_name = "ycbi"

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
        transforms=None,
        start_frame_idx=0,
        num_mesh_pts=1000,
        convert_pose_to_quat=False,
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
        self.transforms = transforms
        self.convert_pose_to_quat = convert_pose_to_quat

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

        self.color_files = self.color_files[start_frame_idx:]  # to bypass still frames
        self.id_strs = self.id_strs[start_frame_idx:]
        self.gt_pose_files = self.gt_pose_files[start_frame_idx:]

        self.num_mesh_pts = num_mesh_pts
        if ycb_meshes_dir is not None:
            ob_name = YCBINEOAT_VIDEONAME_TO_OBJ[self.get_video_name()]
            mesh_path = f"{ycb_meshes_dir}/{ob_name}/textured_simple.obj"
            load_res = load_mesh(mesh_path)
            self.mesh = load_res["mesh"]
            self.mesh_bbox = copy.deepcopy(np.asarray(load_res["bbox"]))
            self.mesh_diameter = load_res["diameter"]
            self.mesh_pts = torch.tensor(trimesh.sample.sample_surface(self.mesh, num_mesh_pts)[0]).float()

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, i):
        sample = {}

        if self.include_rgb:
            sample["rgb"] = self.get_color(i)
        if self.include_mask:
            sample["mask"] = self.get_mask(i)
        if self.include_depth:
            sample["depth"] = self.get_depth(i)
        if self.include_xyz_map:
            sample["xyz_map"] = self.get_xyz_map(i)
        if self.include_gt_pose:
            sample["pose"] = self.get_gt_pose(i)
        sample["rgb_path"] = self.color_files[i]

        sample["intrinsics"] = self.K

        sample["mesh_pts"] = self.mesh_pts
        sample["mesh_bbox"] = self.mesh_bbox
        sample["mesh_diameter"] = self.mesh_diameter

        sample = process_raw_sample(sample, transforms=self.transforms, convert_pose_to_quat=self.convert_pose_to_quat)

        return sample

    def get_gt_pose(self, i):
        pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
        return pose

    def get_color(self, i):
        return load_color(self.color_files[i], wh=(self.w, self.h))

    def get_mask(self, i):
        return load_mask(self.color_files[i].replace("rgb", "masks"), wh=(self.w, self.h))

    def get_depth(self, i):
        return load_depth(self.color_files[i].replace("rgb", "depth"), wh=(self.w, self.h), zfar=self.zfar)

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map, _ = backproj_depth(depth, self.K, do_flip_xy=True)
        return xyz_map

    def get_occ_mask(self, i):
        # NOTE. these files are not present in the dataset
        hand_mask_file = self.color_files[i].replace("rgb", "masks_hand")
        occ_mask = np.zeros((self.h, self.w), dtype=bool)
        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace("rgb", "masks_hand_right")
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        # occ_mask = cv2.resize(occ_mask.astype(np.uint8), (self.w, self.h), interpolation=cv2.INTER_NEAREST) > 0

        return occ_mask.astype(np.uint8)

    def get_video_name(self):
        return str(self.video_dir).split("/")[-1]


class YCBineoatDatasetBenchmark(YCBineoatDataset):
    def __init__(self, preds_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds_path = Path(preds_path)

    def get_pred_pose(self, i):
        pose = load_pose(self.preds_path / f"{self.id_strs[i]}.txt")
        return pose

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample["pose_pred"] = self.get_pred_pose(i)
        return sample
