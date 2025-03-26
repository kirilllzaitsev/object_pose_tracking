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

import json
import os
from pathlib import Path

import numpy as np
import torch
from pose_tracking.config import logger
from pose_tracking.dataset.tracking_ds import TrackingDataset, TrackingMultiObjDataset
from pose_tracking.utils.geom import (
    backproj_depth,
    bbox_to_8_point_centered,
    get_inv_pose,
)
from pose_tracking.utils.io import load_pose, load_semantic_mask
from pose_tracking.utils.pcl import (
    downsample_pcl_via_subsampling,
    downsample_pcl_via_voxels,
)
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from pose_tracking.utils.trimesh_utils import compute_pts_span


class CustomSimDatasetBase(object):

    def __init__(
        self,
        cam_init_rot=None,
        cam_pose_path=None,
        do_remap_pose_from_isaac=True,
        do_load_bbox_from_metadata=False,
        **kwargs,
    ):

        super(CustomSimDatasetBase, self).__init__(**kwargs)

        self.do_remap_pose_from_isaac = do_remap_pose_from_isaac
        self.do_load_bbox_from_metadata = do_load_bbox_from_metadata

        self.video_dir = kwargs["video_dir"]
        self.cam_pose_path = cam_pose_path
        self.cam_init_rot = cam_init_rot

        if cam_pose_path is None:
            if os.path.exists(f"{self.video_dir}/cam_pose.txt"):
                cam_pose_path = f"{self.video_dir}/cam_pose.txt"

        if cam_pose_path is not None:
            assert cam_init_rot is not None
            self.cam_pose_path = Path(cam_pose_path)
            self.c2w = load_pose(cam_pose_path)
            self.cam_init_rot = quaternion_to_matrix(torch.tensor(cam_init_rot)).numpy()
            self.c2w[:3, :3] = self.c2w[:3, :3] @ self.cam_init_rot
            self.w2c = get_inv_pose(pose=self.c2w)
        else:
            self.c2w = None

    def pose_remap_from_isaac(self, pose):
        rt = self.w2c @ pose
        return rt


class CustomSimDataset(CustomSimDatasetBase, TrackingDataset):

    ds_name = "custom_sim"

    def __init__(
        self,
        *args,
        mesh_path=None,
        obj_id=None,
        use_priv_info=False,
        **kwargs,
    ):

        self.use_priv_info = use_priv_info
        self.obj_id = obj_id

        super().__init__(*args, **kwargs)

        if mesh_path is None:
            for name in ["mesh", self.obj_name]:
                mesh_path_prop = f"{self.video_dir}/mesh/{self.obj_name}/{name}.obj"
                if os.path.exists(mesh_path_prop):
                    mesh_path = mesh_path_prop
                    break
        if mesh_path is not None:
            self.set_up_obj_mesh(mesh_path)

    def get_pose(self, idx):
        pose = load_pose(self.pose_files[idx])
        if self.obj_id is not None:
            pose = pose[self.obj_id]
        if self.do_remap_pose_from_isaac:
            pose = self.pose_remap_from_isaac(pose)
        return pose

    def get_pose_paths(self):
        paths = []
        for idx, path in enumerate(self.color_files):
            path = Path(path.replace("rgb/", "pose/")).parent
            if self.obj_id is None:
                path = path / self.obj_name / f"{self.id_strs[idx]}.txt"
            else:
                path = path / f"{self.id_strs[idx]}.npy"
            paths.append(path)
        return paths


class CustomSimDatasetCube(CustomSimDataset):
    ds_name = "custom_sim_cube"

    def __init__(
        self, obj_name="cube", cam_init_rot=(0.5, -0.5, 0.5, -0.5), do_remap_pose_from_isaac=False, *args, **kwargs
    ):
        super().__init__(
            do_remap_pose_from_isaac=do_remap_pose_from_isaac,
            obj_name=obj_name,
            cam_init_rot=cam_init_rot,
            *args,
            **kwargs,
        )

        if self.use_priv_info:
            video_dir = Path(self.video_dir).parent / "custom_sim_textured_no_occlusion"
            for arg in ["use_priv_info", "video_dir", "include_mask", "do_remap_pose_from_isaac"]:
                if arg in kwargs:
                    kwargs.pop(arg)
            self.ds_no_occ = CustomSimDatasetCube(
                do_remap_pose_from_isaac=do_remap_pose_from_isaac,
                include_mask=True,
                video_dir=video_dir,
                cam_init_rot=cam_init_rot,
                obj_name=obj_name,
                *args,
                **kwargs,
            )

    def augment_sample(self, sample, idx):
        if self.use_priv_info:
            sample_no_occ = self.ds_no_occ[idx]
            depth_no_occ = sample_no_occ["depth"]
            mask = sample_no_occ["mask"]
            obj_depth = (depth_no_occ).squeeze().numpy()
            obj_depth[mask == 0] = 0
            obj_depth_3d, _ = backproj_depth(obj_depth, intrinsics=self.K)
            obj_depth_3d = downsample_pcl_via_voxels(obj_depth_3d, voxel_size=0.01)
            priv = downsample_pcl_via_subsampling(obj_depth_3d, num_pts=256)
            sample["priv"] = priv
        return sample


class CustomSimDatasetIkea(CustomSimDataset):
    ds_name = "custom_sim_ikea"

    def __init__(
        self,
        obj_name="object_0",
        cam_init_rot=(0.0, 1.0, 0.0, 0.0),
        do_load_bbox_from_metadata=True,
        obj_id=0,
        *args,
        **kwargs,
    ):
        super().__init__(
            obj_name=obj_name,
            obj_id=obj_id,
            cam_init_rot=cam_init_rot,
            do_load_bbox_from_metadata=do_load_bbox_from_metadata,
            *args,
            **kwargs,
        )

        metadata_path = f"{self.video_dir}/metadata.json"
        assert os.path.exists(metadata_path)
        self.metadata = json.load(open(metadata_path))
        self.mesh_path_orig = self.metadata[0]["usd_path"]

        if do_load_bbox_from_metadata:
            assert self.metadata is not None, f"metadata not found at {metadata_path}"
            assert len(self.metadata) == 1, len(self.metadata)
            self.mesh_bbox = bbox_to_8_point_centered(bbox=self.metadata[0]["bbox"])
            self.mesh_diameter = compute_pts_span(self.mesh_bbox)

    def augment_sample(self, sample, idx):
        sample["class_id"] = [self.metadata[self.obj_id].get("class_id", 0)]
        return sample


class CustomSimMultiObjDataset(CustomSimDatasetBase, TrackingMultiObjDataset):

    ds_name = "custom_sim_multi_obj"

    def __init__(
        self,
        *args,
        mesh_path=None,
        obj_ids=None,
        obj_names=None,
        **kwargs,
    ):

        self.obj_ids = obj_ids
        self.obj_names = obj_names

        self.metadata_path = f"{kwargs['video_dir']}/metadata.json"
        assert os.path.exists(self.metadata_path)
        self.metadata = json.load(open(self.metadata_path))
        self.obj_ids = list(range(len(self.metadata))) if obj_ids is None else obj_ids
        self.segm_labels_to_id = self.metadata[0]["sim_meta"]["segm_labels_to_id"]

        super().__init__(*args, obj_ids=self.obj_ids, obj_names=obj_names, **kwargs)

        if mesh_path is None:
            mesh_path = f"{self.video_dir}/mesh"
            if not os.path.exists(mesh_path):
                print(f"WARNING: {mesh_path} not found")
                mesh_path = None
        if mesh_path is not None:
            self.set_up_obj_mesh(mesh_path)

    def get_pose(self, idx):
        pose = load_pose(self.pose_files[idx])
        if self.obj_ids is not None:
            pose = np.stack([pose[obj_id] for obj_id in self.obj_ids])
        if self.do_remap_pose_from_isaac:
            pose = self.pose_remap_from_isaac(pose)
        return pose

    def pose_remap_from_isaac(self, pose):
        rt = self.w2c @ (pose[..., None] if len(pose.shape) == 2 else pose)
        return rt.squeeze()

    def get_pose_paths(self):
        paths = []
        for idx, path in enumerate(self.color_files):
            path = Path(path.replace("rgb/", "pose/")).parent / f"{self.id_strs[idx]}.npy"
            paths.append(path)
        return paths

    def get_mask(self, i):
        ignored_colors = (
            self.segm_labels_to_id["background"],
            self.segm_labels_to_id["ground"],
            self.segm_labels_to_id["table"],
        )
        return load_semantic_mask(
            self.color_files[i].replace("rgb", "semantic_segmentation"),
            wh=(self.w, self.h),
            ignored_colors=ignored_colors,
        )


class CustomSimMultiObjDatasetIkea(CustomSimMultiObjDataset):
    ds_name = "custom_sim_multi_obj_ikea"

    def __init__(
        self,
        obj_names=["object_0", "object_1", "object_2"],
        obj_ids=None,
        cam_init_rot=(0.0, 1.0, 0.0, 0.0),
        do_load_bbox_from_metadata=True,
        *args,
        **kwargs,
    ):

        super().__init__(
            obj_names=obj_names,
            obj_ids=obj_ids,
            cam_init_rot=cam_init_rot,
            do_load_bbox_from_metadata=do_load_bbox_from_metadata,
            *args,
            **kwargs,
        )

        self.mesh_paths_orig = [x["usd_path"] for x in self.metadata_obj]
        if do_load_bbox_from_metadata:
            assert self.metadata_obj is not None, f"metadata_obj not found at {self.metadata_path}"
            self.mesh_bbox = np.stack(
                [bbox_to_8_point_centered(bbox=self.metadata_obj[oidx]["bbox"]) for oidx in self.obj_ids]
            )
            self.mesh_diameter = [compute_pts_span(self.mesh_bbox[oidx]) for oidx in self.obj_ids]

    def augment_sample(self, sample, idx):
        sample["class_id"] = [self.metadata_obj[oidx].get("class_id", 0) for oidx in self.obj_ids]
        return sample
