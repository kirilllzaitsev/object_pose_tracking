import copy
import json
import os
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from pose_tracking.config import YCBV_SCENE_DIR
from pose_tracking.dataset.ds_meta import (
    YCBINEOAT_VIDEONAME_TO_OBJ,
    YCBV_OBJ_ID_TO_NAME,
    YCBV_OBJ_NAME_TO_COLOR,
    YCBV_OBJ_NAME_TO_ID,
    get_ycb_class_id_from_obj_name,
)
from pose_tracking.dataset.tracking_ds import TrackingDataset, TrackingMultiObjDataset
from pose_tracking.utils.common import get_ordered_paths
from pose_tracking.utils.geom import backproj_depth
from pose_tracking.utils.io import load_mask, load_pose


class YCBvDataset(TrackingMultiObjDataset):

    ds_name = "ycbv"

    def __init__(
        self,
        *args,
        ycb_meshes_dir=f"{YCBV_SCENE_DIR}/models",
        **kwargs,
    ):
        self.resize = 1
        self.K_table = {}
        video_dir = kwargs["video_dir"]
        with open(f"{video_dir}/scene_camera.json", "r") as ff:
            info = json.load(ff)
        for k in info:
            self.K_table[f"{int(k):06d}"] = np.array(info[k]["cam_K"]).reshape(3, 3)
            self.bop_depth_scale = info[k]["depth_scale"]

        if os.path.exists(f"{video_dir}/scene_gt.json"):
            with open(f"{video_dir}/scene_gt.json", "r") as ff:
                self.scene_gt = json.load(ff)
            self.scene_gt = copy.deepcopy(self.scene_gt)  # Release file handle to be pickle-able by joblib
        else:
            self.scene_gt = None
        self.color_files = get_ordered_paths(f"{video_dir}/rgb/*")[:1]
        self.ycb_meshes_dir = ycb_meshes_dir

        self.obj_id_to_names = {}
        self.name_to_ob_id = {}
        self.first_idx = int(Path(self.color_files[0]).stem)

        self.is_synt = "synt" in str(video_dir)
        if self.is_synt:
            self.obj_ids = list(YCBV_OBJ_ID_TO_NAME.keys())
            self.obj_names = list(YCBV_OBJ_ID_TO_NAME.values())
        else:
            self.obj_ids = self.get_instance_ids_in_image(0)
            self.obj_names = [YCBV_OBJ_ID_TO_NAME[k] for k in self.obj_ids]
        for i, ob_id in enumerate(self.obj_ids):
            self.obj_id_to_names[ob_id] = self.obj_names[i]
            self.name_to_ob_id[self.obj_names[i]] = ob_id

        super().__init__(
            *args,
            obj_ids=self.obj_ids,
            obj_names=self.obj_names,
            segm_labels_to_color=YCBV_OBJ_NAME_TO_COLOR,
            pose_dirname="annotated_poses",
            **kwargs,
        )
        self.scene_gt = {
            k: v
            for k, v in self.scene_gt.items()
            if self.start_frame_idx + self.first_idx <= int(k) < self.end_frame_idx + self.first_idx
        }

        if ycb_meshes_dir is not None:
            mesh_paths_obj = [f"{ycb_meshes_dir}/obj_{oid:06d}.ply" for oid in self.obj_ids]
            self.set_up_obj_mesh(mesh_paths_obj, is_mm=True)
            self.obj_id_to_mesh_idx = {oid: midx for midx, oid in enumerate(self.obj_ids)}

    def augment_sample(self, sample, idx):
        # todo: ensure obj_id=class_id
        # synt is single-frame -> doesn't matter if track_ids correspond to class_ids
        sample["class_id"] = self.get_instance_ids_in_image(idx) if self.is_synt else self.obj_ids
        sample["class_name"] = [self.obj_id_to_names[oid] for oid in sample["class_id"]]

        return sample

    def get_mask(self, i):
        mask_path_template = self.color_files[i].replace("rgb", "mask").replace(".png", "_{i:06d}.png")
        annot_idx_to_obj_name = {}
        mask_paths = []
        gt = self.scene_gt[str(i + self.first_idx)]
        if self.is_synt:
            obj_ids = self.get_instance_ids_in_image(i)
            obj_names = [YCBV_OBJ_ID_TO_NAME[k] for k in obj_ids]
            obj_id_to_names = {k: v for k, v in zip(obj_ids, obj_names)}
        else:
            obj_ids = self.obj_ids
            obj_names = self.obj_names
            obj_id_to_names = self.obj_id_to_names
        for i_k, k in enumerate(gt):
            annot_idx_to_obj_name[i_k] = obj_id_to_names[k["obj_id"]]
            mask_paths.append((i_k, mask_path_template.format(i=i_k)))
        mask = np.zeros((self.h, self.w, 3))
        for i_k, mask_path in mask_paths:
            mask_bin = load_mask(mask_path, wh=(self.w, self.h))
            color = self.segm_labels_to_color[annot_idx_to_obj_name[i_k]]
            mask_obj = mask_bin[..., None].repeat(3, axis=-1) * color
            mask += mask_obj
        mask = mask.astype(np.uint8)
        return mask

    def add_mesh_data_to_sample(self, i, sample):
        if self.is_synt:
            obj_ids = self.get_instance_ids_in_image(i)
            mesh_idxs = [self.obj_id_to_mesh_idx[oid] for oid in obj_ids]
            sample["mesh_pts"] = self.mesh_pts[mesh_idxs]
            sample["mesh_bbox"] = self.mesh_bbox[mesh_idxs]
            sample["mesh_diameter"] = self.mesh_diameter[mesh_idxs]
        else:
            return super().add_mesh_data_to_sample(i, sample)
        return sample

    def get_visibility(self, mask, i):
        if self.is_synt:
            visibilities = []
            obj_ids = self.get_instance_ids_in_image(i)
            obj_names = [YCBV_OBJ_ID_TO_NAME[k] for k in obj_ids]
            for oname in obj_names:
                ocolor = self.segm_labels_to_color[oname]
                visibilities.append((mask == ocolor).all(axis=-1).sum() > self.min_pixels_for_visibility)
            return visibilities
        return super().get_visibility(mask)

    def get_gt_poses(self, i_frame):
        gt_poses = []
        name = int(self.id_strs[i_frame])
        for i_k, k in enumerate(self.scene_gt[str(name)]):
            cur = np.eye(4)
            cur[:3, :3] = np.array(k["cam_R_m2c"]).reshape(3, 3)
            cur[:3, 3] = np.array(k["cam_t_m2c"]) / 1e3
            gt_poses.append(cur)
        return np.asarray(gt_poses).reshape(-1, 4, 4)

    def get_pose(self, idx):
        return self.get_gt_poses(idx)

    def get_gt_pose(self, i_frame: int, ob_id, mask=None):
        ob_in_cam = np.eye(4)
        best_iou = -np.inf
        best_gt_mask = None
        name = int(self.id_strs[i_frame])
        for i_k, k in enumerate(self.scene_gt[str(name)]):
            if k["obj_id"] == ob_id:
                cur = np.eye(4)
                cur[:3, :3] = np.array(k["cam_R_m2c"]).reshape(3, 3)
                cur[:3, 3] = np.array(k["cam_t_m2c"]) / 1e3
                if mask is not None:  # When multi-instance exists, use mask to determine which one
                    gt_mask = cv2.imread(
                        f"{self.video_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.png",
                        -1,
                    ).astype(bool)
                    intersect = (gt_mask * mask).astype(bool)
                    union = (gt_mask + mask).astype(bool)
                    iou = float(intersect.sum()) / union.sum()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_mask = gt_mask
                        ob_in_cam = cur
                else:
                    ob_in_cam = cur
                    break
        return ob_in_cam

    def get_intrinsics(self, idx):
        return self.get_K(idx)

    def get_K(self, i_frame):
        K = self.K_table[self.id_strs[i_frame]]
        if self.resize != 1:
            K[:2, :2] *= self.resize
        return K

    def get_video_name(self):
        return str(self.video_dir).split("/")[-1]

    def get_instance_ids_in_image(self, i_frame: int):
        if self.scene_gt is not None:
            ob_ids = []
            for k in self.scene_gt[str(self.first_idx + i_frame)]:
                ob_ids.append(k["obj_id"])
            ob_ids = np.asarray(ob_ids)
        elif self.scene_ob_ids_dict is not None:
            return np.array(self.scene_ob_ids_dict[self.first_idx + self.id_strs[i_frame]])
        else:
            raise RuntimeError(f"{i_frame=} cannot be processed")
        return ob_ids
