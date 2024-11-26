import os
from pathlib import Path

import cv2
import numpy as np
from pose_tracking.dataset.ds_meta import YCBINEOAT_VIDEONAME_TO_OBJ
from pose_tracking.dataset.tracking_ds import TrackingDataset
from pose_tracking.utils.common import get_ordered_paths
from pose_tracking.utils.geom import backproj_depth
from pose_tracking.utils.io import load_pose

try:
    from pizza.lib import image_utils
except ImportError:
    ...


class YCBineoatDataset(TrackingDataset):

    ds_name = "ycbi"

    def __init__(
        self,
        *args,
        include_xyz_map=False,
        include_occ_mask=False,
        ycb_meshes_dir=None,
        **kwargs,
    ):
        super().__init__(*args, pose_dirname="annotated_poses", **kwargs)
        self.ycb_meshes_dir = ycb_meshes_dir
        self.include_xyz_map = include_xyz_map
        self.include_occ_mask = include_occ_mask

        if ycb_meshes_dir is not None:
            self.obj_name = YCBINEOAT_VIDEONAME_TO_OBJ[self.get_video_name()]
            mesh_path = f"{ycb_meshes_dir}/{self.obj_name}/textured_simple.obj"
            self.set_up_obj_mesh(mesh_path)

    def augment_sample(self, sample, idx):
        if self.include_xyz_map:
            sample["xyz_map"] = self.get_xyz_map(idx)

        if self.include_occ_mask:
            sample["occ_mask"] = self.get_occ_mask(idx)

        return sample

    def get_pose_paths(self):
        paths = get_ordered_paths(f"{self.video_dir}/{self.pose_dirname}")
        return paths

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
