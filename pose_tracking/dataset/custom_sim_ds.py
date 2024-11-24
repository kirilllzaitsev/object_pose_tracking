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

import copy
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from pose_tracking.config import logger
from pose_tracking.dataset.ds_common import process_raw_sample
from pose_tracking.utils.geom import backproj_depth, get_inv_pose
from pose_tracking.utils.io import load_color, load_depth, load_mask, load_pose
from pose_tracking.utils.pcl import (
    downsample_pcl_via_subsampling,
    downsample_pcl_via_voxels,
)
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from pose_tracking.utils.segm_utils import mask_erode
from pose_tracking.utils.trimesh_utils import load_mesh
from torch.utils.data import Dataset


class CustomSimDataset(Dataset):

    ds_name = "custom_sim"

    def __init__(
        self,
        root_dir,
        cam_init_rot=(0.5, -0.5, 0.5, -0.5),
        include_masks=False,
        zfar=np.inf,
        transforms=None,
        mesh_path=None,
        cam_pose_path=None,
        do_remap_pose_from_isaac=False,
        do_erode_mask=False,
        use_priv_info=False,
        obj_name="cube",
        convert_pose_to_quat=False,
        num_mesh_pts=2000,
    ):
        self.include_masks = include_masks
        self.do_remap_pose_from_isaac = do_remap_pose_from_isaac
        self.do_erode_mask = do_erode_mask
        self.use_priv_info = use_priv_info
        self.convert_pose_to_quat = convert_pose_to_quat
        self.root_dir = root_dir
        self.transforms = transforms
        self.obj_name = obj_name

        self.color_files = sorted(glob.glob(f"{self.root_dir}/rgb/*.png"))
        self.K = np.loadtxt(f"{self.root_dir}/intrinsics.txt").reshape(3, 3)
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".png", "")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = self.load_mask(self.color_files[0])
        self.zfar = zfar

        if self.use_priv_info:
            self.ds_no_occ = CustomSimDataset(
                root_dir=Path(root_dir).parent / "custom_sim_textured_no_occlusion",
                include_masks=True,
                zfar=zfar,
                transforms=transforms,
                mesh_path=mesh_path,
                cam_pose_path=cam_pose_path,
                do_remap_pose_from_isaac=do_remap_pose_from_isaac,
                do_erode_mask=do_erode_mask,
                use_priv_info=False,
            )

        if cam_pose_path is None:
            if os.path.exists(f"{self.root_dir}/cam_pose.txt"):
                cam_pose_path = f"{self.root_dir}/cam_pose.txt"

        if cam_pose_path is not None:
            cam_pose_path = Path(cam_pose_path)
            self.cam_pose = load_pose(cam_pose_path)
            cam_init_rot = quaternion_to_matrix(torch.tensor(cam_init_rot)).numpy()
            self.cam_pose[:3, :3] = cam_init_rot
            self.w2c = get_inv_pose(pose=self.cam_pose)

        else:
            self.cam_pose = None

        self.mesh_path = mesh_path
        if mesh_path is not None:
            load_res = load_mesh(mesh_path)
            self.mesh = load_res["mesh"]
            self.mesh_bbox = copy.deepcopy(np.asarray(load_res["bbox"]))
            self.mesh_diameter = load_res["diameter"]
            self.mesh_pts = torch.tensor(trimesh.sample.sample_surface(self.mesh, num_mesh_pts)[0]).float()

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        path = self.color_files[idx]
        color = self.get_color(idx)
        depth_raw = self.get_depth(idx)
        pose = self.load_pose(
            Path(self.color_files[idx].replace("rgb/", "pose/")).parent / self.obj_name / f"{self.id_strs[idx]}.txt"
        )

        priv = None
        if self.use_priv_info:
            sample_no_occ = self.ds_no_occ[idx]
            depth_no_occ = sample_no_occ["depth"]
            mask = sample_no_occ["mask"]
            obj_depth = (depth_no_occ).squeeze().numpy()
            obj_depth[mask == 0] = 0
            obj_depth_3d, _ = backproj_depth(obj_depth, intrinsics=self.K)
            obj_depth_3d = downsample_pcl_via_voxels(obj_depth_3d, voxel_size=0.01)
            priv = downsample_pcl_via_subsampling(obj_depth_3d, num_pts=256)

        sample = {}

        if self.include_masks:
            bin_mask = self.load_mask(path)
            if self.do_erode_mask:
                bin_mask = mask_erode(bin_mask, kernel_size=11)
            sample["mask"] = bin_mask

        sample["rgb"] = color
        sample["depth"] = depth_raw
        sample["rgb_path"] = path
        sample["pose"] = pose

        sample["intrinsics"] = self.K

        sample["mesh_pts"] = self.mesh_pts
        sample["mesh_bbox"] = self.mesh_bbox
        sample["mesh_diameter"] = self.mesh_diameter
        sample["obj_name"] = self.obj_name
        sample["priv"] = priv

        sample = process_raw_sample(
            sample, transforms_rgb=self.transforms, convert_pose_to_quat=self.convert_pose_to_quat
        )

        return sample

    def load_pose(self, path):
        pose = load_pose(path)
        if self.do_remap_pose_from_isaac:
            pose = self.pose_remap_from_isaac(pose)
        return pose

    def load_mask(self, path):
        mask = load_mask(path.replace("rgb/", "masks/"))
        mask = torch.from_numpy(mask).float()
        return mask

    def get_color(self, i):
        color = load_color(self.color_files[i], wh=(self.W, self.H))
        return color

    def get_depth(self, i):
        depth = load_depth(self.color_files[i].replace("rgb/", "depth/"), wh=(self.W, self.H), zfar=self.zfar)
        return depth

    def pose_remap_from_isaac(self, pose):
        rt = self.w2c @ pose
        return rt


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

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        sample["pose_pred"] = self.load_pose(f"{self.preds_path}/{self.id_strs[i]}.txt")
        return sample
