import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from pose_tracking.config import logger
from pose_tracking.dataset.ds_common import process_raw_sample
from pose_tracking.metrics import normalize_rotation_matrix
from pose_tracking.utils.common import get_ordered_paths
from pose_tracking.utils.geom import (
    convert_3d_bbox_to_2d,
    interpolate_bbox_edges,
    world_to_2d,
)
from pose_tracking.utils.io import load_color, load_depth, load_mask, load_pose
from pose_tracking.utils.misc import print_cls
from pose_tracking.utils.segm_utils import infer_bounding_box, mask_morph
from pose_tracking.utils.trimesh_utils import load_mesh
from torch.utils.data import Dataset


class TrackingDataset(Dataset):

    ds_name = None

    def __init__(
        self,
        video_dir,
        pose_dirname="pose",
        downscale=1,
        obj_name=None,
        shorter_side=None,
        zfar=10,
        include_rgb=True,
        include_depth=True,
        include_mask=True,
        include_pose=True,
        include_bbox_2d=False,
        do_erode_mask=False,
        do_convert_depth_to_m=True,
        do_normalize_bbox=False,
        do_normalize_depth=False,
        max_depth=10,
        bbox_format="xyxy",
        transforms_rgb=None,
        start_frame_idx=0,
        end_frame_idx=None,
        num_mesh_pts=2000,
        mask_pixels_prob=0.0,
        rgb_file_extension="png",
        color_file_id_strs=None,
        rot_repr="quaternion",
        t_repr="3d",
    ):
        self.include_rgb = include_rgb
        self.include_mask = include_mask
        self.include_depth = include_depth
        self.include_pose = include_pose
        self.include_bbox_2d = include_bbox_2d
        self.do_erode_mask = do_erode_mask
        self.do_convert_depth_to_m = do_convert_depth_to_m
        self.do_normalize_bbox = do_normalize_bbox
        self.do_normalize_depth = do_normalize_depth

        self.video_dir = video_dir
        self.obj_name = obj_name
        self.downscale = downscale
        self.zfar = zfar
        self.transforms_rgb = transforms_rgb
        self.mask_pixels_prob = mask_pixels_prob
        self.num_mesh_pts = num_mesh_pts
        self.start_frame_idx = start_frame_idx
        self.pose_dirname = pose_dirname
        self.bbox_format = bbox_format
        self.max_depth = max_depth
        self.color_file_id_strs = color_file_id_strs
        self.rot_repr = rot_repr
        self.t_repr = t_repr

        self.color_files = get_ordered_paths(f"{self.video_dir}/rgb/*.{rgb_file_extension}")
        if color_file_id_strs is not None:
            self.color_files = [f for f in self.color_files if Path(f).stem in color_file_id_strs]
        self.end_frame_idx = end_frame_idx or len(self.color_files)
        self.color_files = self.color_files[self.start_frame_idx : self.end_frame_idx]
        self.num_frames = len(self.color_files)

        self.id_strs = [Path(f).stem for f in self.color_files]
        self.pose_files = self.get_pose_paths()

        self.mesh = None
        self.h, self.w = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = self.get_mask(0)
        self.t_dim = 3 if t_repr == "3d" else 2

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.h, self.w)

        self.h = int(self.h * self.downscale)
        self.w = int(self.w * self.downscale)

        intrinsics_path = f"{video_dir}/cam_K.txt"
        if not os.path.exists(intrinsics_path):
            intrinsics_path = f"{video_dir}/intrinsics.txt"
            assert os.path.exists(intrinsics_path), f"Could not find intrinsics file at {intrinsics_path}"

        self.K = np.loadtxt(intrinsics_path).reshape(3, 3)
        self.K[:2] *= self.downscale

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        sample = {}

        if self.include_rgb:
            sample["rgb"] = self.get_color(i)

        if self.include_mask:
            mask = self.get_mask(i)
            if self.do_erode_mask:
                mask = mask_morph(mask, kernel_size=11)
            sample["mask"] = mask

        if self.include_depth:
            depth = self.get_depth(i)
            if self.do_normalize_depth:
                depth /= self.max_depth
                sample["max_depth"] = self.max_depth
            sample["depth"] = depth

        if self.include_pose:
            sample["pose"] = self.get_pose(i)
            # sample["pose"][:3, :3] = normalize_rotation_matrix(sample["pose"][:3, :3])

        sample["rgb_path"] = self.color_files[i]
        sample["intrinsics"] = self.K
        sample["obj_name"] = self.obj_name

        if self.mesh is not None:
            sample["mesh_pts"] = self.mesh_pts
            sample["mesh_bbox"] = self.mesh_bbox
            sample["mesh_diameter"] = self.mesh_diameter

        if self.include_bbox_2d:
            if not self.include_mask:
                mask = self.get_mask(i)
                if self.do_erode_mask:
                    mask = mask_morph(mask, kernel_size=11)
            bbox_2d = infer_bounding_box(mask)
            if bbox_2d is None:
                logger.error(f"Could not infer bbox for {self.color_files[i]}")
                return None
            bbox_2d = bbox_2d.astype(np.float32)
            if self.do_normalize_bbox:
                bbox_2d[:, 0] /= self.w
                bbox_2d[:, 1] /= self.h
            if self.bbox_format == "xyxy":
                bbox_2d = bbox_2d.reshape(1, -1)
                sample["bbox_2d"] = bbox_2d
            elif self.bbox_format == "cxcywh":
                bbox_2d = bbox_2d.reshape(-1)
                sample["bbox_2d"] = np.array(
                    [
                        (bbox_2d[0] + bbox_2d[2]) / 2,
                        (bbox_2d[1] + bbox_2d[3]) / 2,
                        bbox_2d[2] - bbox_2d[0],
                        bbox_2d[3] - bbox_2d[1],
                    ]
                )
            ibbs_res = interpolate_bbox_edges(self.mesh_bbox, num_points=24)
            sample["bbox_2d_kpts"] = world_to_2d(ibbs_res["all_points"], K=self.K, rt=sample["pose"])
            # normalize bbox_2d_kpts to [0, 1]
            sample["bbox_2d_kpts"] /= np.array([self.w, self.h])
            sample["bbox_2d_kpts_collinear_idxs"] = ibbs_res["collinear_quad_idxs"]

        sample = self.augment_sample(sample, i)

        sample = process_raw_sample(
            sample,
            transforms_rgb=self.transforms_rgb,
            mask_pixels_prob=self.mask_pixels_prob,
            rot_repr=self.rot_repr,
            t_repr=self.t_repr,
        )

        return sample

    def augment_sample(self, sample, idx):
        return sample

    def get_pose_paths(self):
        return [f"{self.video_dir}/{self.pose_dirname}/{id_str}.txt" for id_str in self.id_strs]

    def get_color(self, i):
        return load_color(self.color_files[i], wh=(self.w, self.h))

    def get_mask(self, i):
        return load_mask(self.color_files[i].replace("rgb", "masks"), wh=(self.w, self.h))

    def get_depth(self, i):
        return load_depth(
            self.color_files[i].replace("rgb", "depth"),
            wh=(self.w, self.h),
            zfar=self.zfar,
            do_convert_to_m=self.do_convert_depth_to_m,
        )

    def get_pose(self, idx):
        return load_pose(self.pose_files[idx])

    def set_up_obj_mesh(self, mesh_path):
        load_res = load_mesh(mesh_path)
        self.mesh = load_res["mesh"]
        self.mesh_bbox = copy.deepcopy(np.asarray(load_res["bbox"]))
        self.mesh_diameter = load_res["diameter"]
        self.mesh_pts = torch.tensor(trimesh.sample.sample_surface(self.mesh, self.num_mesh_pts)[0]).float()
        self.mesh_path = mesh_path
        self.mesh_path_orig = mesh_path

    def __repr__(self):
        return print_cls(self, excluded_attrs=["color_files", "pose_files", "id_strs", "init_mask", "mesh_pts"])


class TrackingDatasetEval:
    def __init__(self, ds, preds_path):
        self.ds = ds
        self.preds_path = Path(preds_path)
        self.pred_file_paths = list(sorted(self.preds_path.glob("*.txt")))
        if len(self.pred_file_paths) != len(self.ds):
            logger.warning(
                f"Number of predictions ({len(self.pred_file_paths)}) does not match number of samples ({len(self.ds)})"
            )

    def __len__(self):
        return min(len(self.ds), len(self.pred_file_paths))

    def __getitem__(self, idx):
        sample = self.ds[idx]
        sample["pose_pred"] = load_pose(f"{self.preds_path}/{self.ds.id_strs[idx]}.txt")
        return sample
