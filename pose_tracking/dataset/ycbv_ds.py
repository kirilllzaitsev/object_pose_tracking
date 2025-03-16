import copy
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from pose_tracking.dataset.ds_meta import (
    YCBINEOAT_VIDEONAME_TO_OBJ,
    YCBV_OBJ_ID_TO_NAME,
    YCBV_OBJ_NAME_TO_ID,
    get_ycb_class_id_from_obj_name,
)
from pose_tracking.dataset.tracking_ds import TrackingDataset
from pose_tracking.utils.common import get_ordered_paths
from pose_tracking.utils.geom import backproj_depth
from pose_tracking.utils.io import load_mask, load_pose


class YCBvDataset(TrackingDataset):

    ds_name = "ycbv"

    def __init__(
        self,
        *args,
        include_xyz_map=False,
        include_occ_mask=False,
        ycb_meshes_dir="/media/master/t7/msc_studies/pose_estimation/object_pose_tracking/data/ycbv/models",
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

        super().__init__(*args, pose_dirname="annotated_poses", **kwargs)
        if os.path.exists(f"{video_dir}/scene_gt.json"):
            with open(f"{video_dir}/scene_gt.json", "r") as ff:
                self.scene_gt = json.load(ff)
            self.scene_gt = copy.deepcopy(self.scene_gt)  # Release file handle to be pickle-able by joblib
            assert len(self.scene_gt) == len(self.color_files)
        else:
            self.scene_gt = None
        self.ycb_meshes_dir = ycb_meshes_dir

        self.ob_id_to_names = {}
        self.name_to_ob_id = {}
        # self.ob_ids = np.arange(1, 22).astype(int).tolist()
        # TODO
        self.ob_ids = self.get_instance_ids_in_image(0)[:1]
        names = [YCBV_OBJ_ID_TO_NAME[k] for k in self.ob_ids]
        for i, ob_id in enumerate(self.ob_ids):
            self.ob_id_to_names[ob_id] = names[i]
            self.name_to_ob_id[names[i]] = ob_id

        self.obj_name = names[0]
        self.class_id = self.ob_ids[0]

        if ycb_meshes_dir is not None:
            mesh_path = f"{ycb_meshes_dir}/obj_{self.class_id:06d}.ply"
            self.set_up_obj_mesh(mesh_path, is_mm=True)

    def augment_sample(self, sample, idx):
        sample["class_id"] = [self.class_id]

        return sample

    def get_mask(self, i):
        # TODO: npy  for masks
        return load_mask(self.color_files[i].replace("rgb", "mask").replace(".png", "_000000.png"), wh=(self.w, self.h))

    def get_gt_poses(self, i_frame, ob_id):
        gt_poses = []
        name = int(self.id_strs[i_frame])
        for i_k, k in enumerate(self.scene_gt[str(name)]):
            if k["obj_id"] == ob_id:
                cur = np.eye(4)
                cur[:3, :3] = np.array(k["cam_R_m2c"]).reshape(3, 3)
                cur[:3, 3] = np.array(k["cam_t_m2c"]) / 1e3
                gt_poses.append(cur)
        return np.asarray(gt_poses).reshape(-1, 4, 4)

    def get_pose(self, idx):
        i_frame = idx
        return self.get_gt_pose(i_frame, self.ob_ids[0])

    def get_gt_pose(self, i_frame: int, ob_id, mask=None, use_my_correction=False):
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
        ob_ids = []
        if self.scene_gt is not None:
            name = int(os.path.basename(self.color_files[i_frame]).split(".")[0])
            for k in self.scene_gt[str(name)]:
                ob_ids.append(k["obj_id"])
        elif self.scene_ob_ids_dict is not None:
            return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])
        else:
            mask_dir = os.path.dirname(self.color_files[0]).replace("rgb", "mask_visib")
            id_str = self.id_strs[i_frame]
            mask_files = sorted(glob.glob(f"{mask_dir}/{id_str}_*.png"))
            ob_ids = []
            for mask_file in mask_files:
                ob_id = int(os.path.basename(mask_file).split(".")[0].split("_")[1])
                ob_ids.append(ob_id)
        ob_ids = np.asarray(ob_ids)
        return ob_ids
