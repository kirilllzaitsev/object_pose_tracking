import os
import re
from pathlib import Path

import cv2
import numpy as np
from pose_tracking.dataset.ds_meta import (
    YCBINEOAT_VIDEONAME_TO_OBJ,
    YCBV_OBJ_NAME_TO_ID,
)
from pose_tracking.dataset.tracking_ds import TrackingDataset
from pose_tracking.utils.common import get_ordered_paths
from pose_tracking.utils.geom import backproj_depth
from pose_tracking.utils.io import load_pose

try:
    from pizza.lib.dataloader import augmentation, image_utils, utils
    from pizza.lib.dataset.modelNet import dataloader_utils
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
        # TMP: the mask labels are especially bad for these videos, not usable for obj det
        # because inferring bbox based on mask
        if "bleach0" in str(kwargs["video_dir"]):
            kwargs["end_frame_idx"] = -2
        elif "cracker_box_yalehand0" in str(kwargs["video_dir"]):
            kwargs["end_frame_idx"] = -80
        elif "sugar_box1" in str(kwargs["video_dir"]):
            kwargs["end_frame_idx"] = 840

        super().__init__(*args, pose_dirname="annotated_poses", **kwargs)
        self.ycb_meshes_dir = ycb_meshes_dir
        self.include_xyz_map = include_xyz_map
        self.include_occ_mask = include_occ_mask

        self.obj_name = YCBINEOAT_VIDEONAME_TO_OBJ[self.get_video_name()]
        obj_name_no_pref = re.search("\d+_(.*)", self.obj_name)
        if obj_name_no_pref is None:
            raise ValueError(f"Could not extract object name from {self.obj_name}")
        self.class_id = YCBV_OBJ_NAME_TO_ID[obj_name_no_pref.group(1)] - 1

        if ycb_meshes_dir is not None:
            mesh_path = f"{ycb_meshes_dir}/{self.obj_name}/textured_simple.obj"
            self.set_up_obj_mesh(mesh_path)

    def augment_sample(self, sample, idx):
        if self.include_xyz_map:
            sample["xyz_map"] = self.get_xyz_map(idx)

        if self.include_occ_mask:
            sample["occ_mask"] = self.get_occ_mask(idx)

        sample["class_id"] = [self.class_id]

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


class YCBineoatDatasetPizza(YCBineoatDataset):
    """
    ratio
    uv_first_frame
    gt_delta_uv
    gt_delta_depth
    depth_first_frame
    gt_delta_rotation
    rotation_first_frame
    gt_rotations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sequence_obj = dataloader_utils.SequenceProcessing(
            root_path="/media/master/t7/msc_studies/pose_estimation/object_pose_tracking/data/ycbineoat/mustard0/rgb",
            intrinsic=self.K,
        )
        sequence_obj.create_list_index_frame_in_sequences()
        sequence_obj.create_list_img_path()
        self.gt_sequence_obj = sequence_obj.create_gt_tracking()
        # shuffle?!
        self.image_size = np.array([self.h, self.w]).max()

    def __len__(self):
        return len(self.gt_sequence_obj["delta_rotation"])

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        # bbox_size = np.max([bbox_sequence[2] - bbox_sequence[0], bbox_sequence[3] - bbox_sequence[1]])
        bbox = sample["bbox_2d"].numpy()[0]
        bbox_size = np.max(bbox[2:] - bbox[:2])
        max_size_with_margin = bbox_size * 1.3  # margin = 0.2 x max_dim
        margin = bbox_size * 0.15
        # bbox_sequence = bbox_sequence + np.array([-margin, -margin, margin, margin])

        bbox_sequence_square = image_utils.make_bbox_square(bbox, max_size_with_margin)
        ratio = self.image_size / max_size_with_margin
        names = [
            "delta_rotation",
            "delta_uv",
            "delta_depth",
            "rotation_first_frame",
            "uv_first_frame",
            "depth_first_frame",
            "gt_rotations",
            "gt_translations",
        ]
        for name in names:
            v = self.gt_sequence_obj[name][i]
            if name == "delta_uv":
                v = v * ratio / (self.image_size / 2)
            elif name == "delta_depth":
                v = v * ratio
            sample[name] = v
        sample["ratio"] = ratio
        sample["bbox_sequence_square"] = bbox_sequence_square
        return sample

    def _fetch_sequence(self, img_path, save_path=None):
        sequence_img, list_bbox = [], []
        for i in range(2):
            img = image_utils.open_image(img_path[i])
            sequence_img.append(img)
            list_bbox.append(np.asarray(img.getbbox()))
        # take max bbox of two images
        bbox_sequence = np.zeros(4)
        bbox_sequence[0] = np.min([list_bbox[0][0], list_bbox[1][0]])
        bbox_sequence[1] = np.min([list_bbox[0][1], list_bbox[1][1]])
        bbox_sequence[2] = np.max([list_bbox[0][2], list_bbox[1][2]])
        bbox_sequence[3] = np.max([list_bbox[0][3], list_bbox[1][3]])

        bbox_size = np.max([bbox_sequence[2] - bbox_sequence[0], bbox_sequence[3] - bbox_sequence[1]])
        max_size_with_margin = bbox_size * 1.3  # margin = 0.2 x max_dim
        margin = bbox_size * 0.15
        bbox_sequence = bbox_sequence + np.array([-margin, -margin, margin, margin])
        bbox_sequence_square = image_utils.make_bbox_square(bbox_sequence, max_size_with_margin)
        ratio = self.image_size / max_size_with_margin  # keep this value to predict translation later
        for i in range(2):
            cropped_img = sequence_img[i].crop(bbox_sequence_square)
            cropped_resized_img = cropped_img.resize((self.image_size, self.image_size))
            sequence_img[i] = cropped_resized_img
        # if "train" in self.split and self.use_augmentation:
        #     sequence_img = augmentation.apply_data_augmentation(2, sequence_img)
        if save_path is None:
            seq_img = np.zeros((2, 3, self.image_size, self.image_size))
            for i in range(2):
                seq_img[i] = image_utils.normalize(sequence_img[i].convert("RGB"))
            return seq_img, ratio, bbox_sequence_square
