import copy
import os
import re
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
    cam_to_2d,
    convert_3d_bbox_to_2d,
    egocentric_to_allocentric,
    interpolate_bbox_edges,
    world_to_2d,
    world_to_cam,
)
from pose_tracking.utils.io import (
    load_color,
    load_depth,
    load_mask,
    load_pose,
    load_semantic_mask,
)
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
        include_bbox_3d=True,
        include_nocs=False,
        do_erode_mask=False,
        do_convert_depth_to_m=True,
        do_normalize_bbox=False,
        do_subtract_bg=False,
        do_normalize_depth=False,
        use_allocentric_pose=False,
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
        is_intrinsics_for_all_samples=True,
        use_mask_for_bbox_2d=True,
        bbox_num_kpts=32,
        dino_features_folder_name=None,
        num_objs=1,
    ):
        if do_subtract_bg:
            assert do_subtract_bg and include_mask, do_subtract_bg

        self.include_rgb = include_rgb
        self.include_mask = include_mask
        self.include_depth = include_depth
        self.include_pose = include_pose
        self.include_bbox_2d = include_bbox_2d
        self.include_bbox_3d = include_bbox_3d
        self.do_erode_mask = do_erode_mask
        self.do_convert_depth_to_m = do_convert_depth_to_m
        self.do_normalize_bbox = do_normalize_bbox
        self.do_normalize_depth = do_normalize_depth
        self.do_subtract_bg = do_subtract_bg
        self.use_allocentric_pose = use_allocentric_pose
        self.include_nocs = include_nocs
        self.use_mask_for_bbox_2d = use_mask_for_bbox_2d

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
        self.bbox_num_kpts = bbox_num_kpts
        self.dino_features_folder_name = dino_features_folder_name
        self.num_objs = num_objs

        self.do_load_dino_features = dino_features_folder_name is not None

        self.color_files = get_ordered_paths(f"{self.video_dir}/rgb/*.{rgb_file_extension}")
        if color_file_id_strs is not None:
            self.color_files = [f for f in self.color_files if Path(f).stem in color_file_id_strs]

        self.id_strs = [Path(f).stem for f in self.color_files]
        self.pose_files = self.get_pose_paths()
        self.end_frame_idx = end_frame_idx or len(self.color_files)
        self.color_files = self.color_files[self.start_frame_idx : self.end_frame_idx]
        self.color_files = np.array(self.color_files)
        self.pose_files = self.pose_files[self.start_frame_idx : self.end_frame_idx]
        self.pose_files = np.array(self.pose_files)
        self.num_frames = len(self.color_files)

        self.mesh = None
        self.h, self.w = cv2.imread(self.color_files[0]).shape[:2]
        self.init_mask = torch.tensor(self.get_mask(0)) if include_mask else None
        self.t_dim = 3 if t_repr == "3d" else 2

        if shorter_side is not None:
            self.downscale = shorter_side / min(self.h, self.w)

        self.h = int(self.h * self.downscale)
        self.w = int(self.w * self.downscale)

        if is_intrinsics_for_all_samples:
            intrinsics_path = f"{video_dir}/cam_K.txt"
            if not os.path.exists(intrinsics_path):
                intrinsics_path = f"{video_dir}/intrinsics.txt"
            if os.path.exists(intrinsics_path):
                self.K = torch.tensor(np.loadtxt(intrinsics_path).reshape(3, 3))
                self.K[:2] *= self.downscale
            else:
                print(f"Could not find intrinsics file at {intrinsics_path}")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        sample = {}

        if self.include_rgb:
            sample["rgb"] = self.get_color(i)

        if self.include_mask or self.num_objs > 1:
            mask = self.get_mask(i)
            is_visible = self.get_visibility(mask)
            sample["is_visible"] = is_visible if self.num_objs > 1 else [is_visible]
            if self.include_mask:
                sample["mask"] = mask
        else:
            sample["is_visible"] = [True]

        if self.include_depth:
            depth = self.get_depth(i)
            if self.do_normalize_depth:
                depth /= self.max_depth
                sample["max_depth"] = self.max_depth
            sample["depth"] = depth

        if self.do_subtract_bg:
            sample["rgb"] = sample["rgb"] * sample["mask"][..., None]
            if self.include_depth:
                sample["depth"] = sample["depth"] * sample["mask"]

        if self.include_nocs:
            ...

        if self.do_load_dino_features:
            features_path = f"{self.video_dir}/{self.dino_features_folder_name}/{self.id_strs[i]}.pt"
            assert os.path.exists(features_path), f"Could not find features at {features_path}"
            if os.path.exists(features_path):
                sample["features_rgb"] = torch.load(features_path, weights_only=False)

        if self.include_pose:
            sample["pose"] = self.get_pose(i)
            if self.use_allocentric_pose:
                sample["pose"] = egocentric_to_allocentric(sample["pose"])

        sample["rgb_path"] = self.color_files[i]
        sample["intrinsics"] = self.get_intrinsics(i)
        sample["obj_name"] = self.obj_name

        if self.mesh is not None:
            sample["mesh_pts"] = self.mesh_pts
            sample["mesh_bbox"] = self.mesh_bbox
            sample["mesh_diameter"] = self.mesh_diameter

        if (self.include_bbox_2d and self.bbox_num_kpts == 32) or self.include_bbox_3d:
            ibbs_res = interpolate_bbox_edges(copy.deepcopy(self.mesh_bbox), num_points=24)
            sample["bbox_2d_kpts_collinear_idxs"] = ibbs_res["collinear_quad_idxs"]
            bbox_3d_kpts = ibbs_res["all_points"]
            if self.include_bbox_3d:
                sample["bbox_3d_kpts"] = world_to_cam(bbox_3d_kpts, sample["pose"]).astype(np.float32)
                sample["bbox_3d_kpts_mesh"] = bbox_3d_kpts.astype(np.float32)
                sample["bbox_3d_kpts_corners"] = world_to_cam(self.mesh_bbox, sample["pose"]).astype(np.float32)
                sample["bbox_3d_kpts_corners"][:, 2] *= 1e-0
        else:
            bbox_3d_kpts = copy.deepcopy(self.mesh_bbox)

        if self.include_bbox_2d:
            if self.use_mask_for_bbox_2d:
                bbox_2ds = []
                if not self.include_mask:
                    mask = self.get_mask(i)
                    if self.do_erode_mask:
                        mask = mask_morph(mask, kernel_size=11)
                if self.num_objs > 1:
                    colors = [x for x in np.unique(mask.reshape(-1, mask.shape[2]), axis=0) if x.sum() > 0]
                    bin_masks = [np.all(mask == c, axis=-1) for c in colors]
                else:
                    bin_masks = [mask]
                for bin_mask in bin_masks:
                    bbox_2d = infer_bounding_box(bin_mask)
                    if bbox_2d is None:
                        print(f"ERROR: Could not infer bbox for {self.color_files[i]}")
                        return None
                    bbox_2ds.append(bbox_2d)
            else:
                bbox_2ds = list(
                    convert_3d_bbox_to_2d(
                        self.mesh_bbox, sample["intrinsics"].numpy(), hw=(self.h, self.w), pose=sample["pose"]
                    )
                )
            for idx, bbox_2d in enumerate(bbox_2ds):
                bbox_2d = bbox_2d.astype(np.float32)
                if self.do_normalize_bbox:
                    bbox_2d[..., 0] /= self.w
                    bbox_2d[..., 1] /= self.h
                if self.bbox_format == "xyxy":
                    bbox_2d = bbox_2d.reshape(1, -1)
                    bbox_2ds[idx] = bbox_2d
                elif self.bbox_format == "cxcywh":
                    bbox_2d = bbox_2d.reshape(-1)
                    bbox_2ds[idx] = np.array(
                        [
                            (bbox_2d[0] + bbox_2d[2]) / 2,
                            (bbox_2d[1] + bbox_2d[3]) / 2,
                            bbox_2d[2] - bbox_2d[0],
                            bbox_2d[3] - bbox_2d[1],
                        ]
                    )
            sample["bbox_2d"] = np.stack(bbox_2ds)

            bbox_3d_kpts = world_to_cam(bbox_3d_kpts, sample["pose"])
            sample["bbox_2d_kpts_depth"] = bbox_3d_kpts[..., 2:]
            sample["bbox_2d_kpts"] = cam_to_2d(bbox_3d_kpts, K=sample["intrinsics"])
            sample["bbox_2d_kpts"] /= np.array([self.w, self.h])

        sample = self.augment_sample(sample, i)

        sample = process_raw_sample(
            sample,
            transforms_rgb=self.transforms_rgb,
            mask_pixels_prob=self.mask_pixels_prob,
            rot_repr=self.rot_repr,
            t_repr=self.t_repr,
        )

        return sample

    def get_intrinsics(self, idx):
        return self.K

    def augment_sample(self, sample, idx):
        return sample

    def get_pose_paths(self):
        return [f"{self.video_dir}/{self.pose_dirname}/{id_str}.txt" for id_str in self.id_strs]

    def get_color(self, i):
        return load_color(self.color_files[i], wh=(self.w, self.h))

    def get_mask(self, i):
        mask = load_mask(self.color_files[i].replace("rgb", "masks"), wh=(self.w, self.h))
        if self.do_erode_mask:
            mask = mask_morph(mask, kernel_size=11)
        return mask

    def get_depth(self, i):
        return load_depth(
            self.color_files[i].replace("rgb", "depth"),
            wh=(self.w, self.h),
            zfar=self.zfar,
            do_convert_to_m=self.do_convert_depth_to_m,
        )

    def get_pose(self, idx):
        return load_pose(self.pose_files[idx])

    def get_visibility(self, mask):
        return mask.sum() > self.min_pixels_for_visibility

    def set_up_obj_mesh(self, mesh_path, is_mm=False):
        load_res = load_mesh(mesh_path, is_mm=is_mm)
        self.mesh = load_res["mesh"]
        self.mesh_bbox = copy.deepcopy(np.asarray(load_res["bbox"]))
        self.mesh_diameter = load_res["diameter"]
        self.mesh_pts = torch.tensor(trimesh.sample.sample_surface(self.mesh, self.num_mesh_pts)[0]).float()
        self.mesh_path = mesh_path
        self.mesh_path_orig = mesh_path

    def __repr__(self):
        return print_cls(
            self,
            excluded_attrs=[
                "color_files",
                "pose_files",
                "id_strs",
                "init_mask",
                "mesh_pts",
                "meta_file",
                "color_file_id_strs",
                "meta_paths",
                "metadata",
            ],
        )


class TrackingMultiObjDataset(TrackingDataset):

    def __init__(self, *args, segm_labels_to_id, obj_ids=None, obj_names=None, **kwargs):
        self.segm_labels_to_id = segm_labels_to_id

        self.obj_ids = obj_ids or []
        self.obj_names = obj_names or []

        super().__init__(*args, **kwargs, num_objs=max(len(self.obj_ids), len(self.obj_names)))

    def set_up_obj_mesh(self, mesh_path, is_mm=False):
        meshes = []
        mesh_bbox = []
        mesh_diameter = []
        mesh_pts = []
        mesh_paths = []
        mesh_paths_orig = []

        for obj_name in self.obj_names:
            obj_mesh_dir = f"{mesh_path}/{obj_name}"
            load_res = load_mesh(f"{obj_mesh_dir}/mesh.obj", is_mm=is_mm)
            mesh = load_res["mesh"]
            meshes.append(mesh)
            mesh_bbox.append(copy.deepcopy(np.asarray(load_res["bbox"])))
            mesh_diameter.append(load_res["diameter"])
            mesh_pts.append(torch.tensor(trimesh.sample.sample_surface(mesh, self.num_mesh_pts)[0]).float())
            mesh_paths.append(mesh_path)
            mesh_paths_orig.append(mesh_path)
        self.mesh = meshes
        self.mesh_bbox = mesh_bbox
        self.mesh_diameter = mesh_diameter
        self.mesh_pts = torch.stack(mesh_pts)
        self.mesh_path = mesh_paths
        # TODO
        self.mesh_path_orig = mesh_paths_orig[0]

    def get_visibility(self, mask):
        visibilities = []
        for oname in self.obj_names:
            ocolor = self.segm_labels_to_id[oname]
            visibilities.append((mask == ocolor).all(axis=-1).sum() > self.min_pixels_for_visibility)
        return visibilities


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
