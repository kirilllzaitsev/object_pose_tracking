import os
import pickle
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from pose_tracking.config import HO3D_ROOT
from pose_tracking.dataset.ds_meta import (
    HO3D_VIDEONAME_TO_OBJ,
    get_ycb_class_id_from_obj_name,
)
from pose_tracking.dataset.tracking_ds import TrackingDataset
from pose_tracking.utils.io import load_depth, load_mask, resize_img


class HO3DDataset(TrackingDataset):

    ds_name = "ho3d"

    def __init__(
        self,
        video_dir,
        *args,
        include_occ_mask=False,
        do_load_mesh=True,
        **kwargs,
    ):
        self.do_load_mesh = do_load_mesh
        self.include_occ_mask = include_occ_mask
        self.use_xmem_masks = os.path.exists(f"{video_dir}/masks_XMem")

        meta_paths = glob(f"{video_dir}/meta/*.pkl")
        color_file_id_strs = []
        for meta_path in meta_paths:
            # if file size less than 1 kb, discard
            if os.path.getsize(meta_path) < 1e3:
                continue
            color_file_id_strs.append(Path(meta_path).stem)
        # print(f"Taking {len(color_file_id_strs)}/{len(meta_paths)} frames from {Path(video_dir).name}")

        super().__init__(
            *args, video_dir=video_dir, rgb_file_extension="jpg", color_file_id_strs=color_file_id_strs, **kwargs
        )

        self.meta_file_path = self.color_files[0].replace(".jpg", ".pkl").replace("rgb", "meta")
        self.meta_file = pickle.load(open(self.meta_file_path, "rb"))
        self.K = self.meta_file["camMat"]
        self.K[:2] *= self.downscale
        self.obj_name = self.meta_file["objName"]

        self.glcam_in_cvcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.video_name = self.get_video_name()
        self.class_id = get_ycb_class_id_from_obj_name(self.obj_name)

        if do_load_mesh:
            self.mesh_path = f"{HO3D_ROOT}/models/{self.obj_name}/textured_simple.obj"
            self.set_up_obj_mesh(self.mesh_path)

    def get_video_name(self):
        return os.path.dirname(os.path.abspath(self.color_files[0])).split("/")[-2]

    def get_mask(self, i):
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        if self.use_xmem_masks:
            video_name = self.get_video_name()
            path = f"{HO3D_ROOT}/masks_XMem/{video_name}/{index:05d}.png"
        else:
            path = self.color_files[i].replace("rgb", "seg").replace(".jpg", ".png")
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        obj_color = (0, 255, 0)
        # hand_color = (255, 0, 0)

        mask_obj = mask == obj_color
        mask[~mask_obj] = 0

        green_color = (0, 255, 0)
        white_color = (255, 255, 255)
        is_green = np.all(mask == green_color, axis=-1)
        mask[is_green] = white_color

        mask = resize_img(mask, wh=(self.w, self.h))

        mask = mask[..., 0]

        return mask

    def get_occ_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        if self.use_xmem_masks:
            path = f"{HO3D_ROOT}/masks_XMem/{video_name}_hand/{index:04d}.png"
        else:
            path = self.color_files[i].replace("rgb", "masks_hand")
        mask = load_mask(path)
        return mask

    def get_depth(self, i):
        depth_scale = 0.00012498664727900177
        depth = load_depth(self.color_files[i].replace(".jpg", ".png").replace("rgb", "depth"), do_convert_to_m=False)
        depth = (depth[..., 2] + depth[..., 1] * 256) * depth_scale
        return depth

    def get_pose(self, idx):
        meta_file = self.color_files[idx].replace(".jpg", ".pkl").replace("rgb", "meta")
        meta = pickle.load(open(meta_file, "rb"))
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = self.glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt

    def augment_sample(self, sample, idx):
        if self.include_occ_mask:
            sample["occ_mask"] = self.get_occ_mask(idx)

        sample["class_id"] = [self.class_id]

        return sample
