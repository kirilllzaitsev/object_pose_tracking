import os
import pickle
from glob import glob

import cv2
import numpy as np
import torch
from pose_tracking.config import HO3D_ROOT
from pose_tracking.dataset.ds_common import get_ds_sample
from pose_tracking.dataset.ds_meta import HO3D_VIDEONAME_TO_OBJ
from pose_tracking.utils.io import load_color, load_depth, load_mask
from pose_tracking.utils.trimesh_utils import load_mesh


class HO3DDataset(torch.utils.data.Dataset):

    ds_name = "ho3d"

    def __init__(self, video_dir, transforms=None, include_masks=True, include_occ_masks=False, do_load_mesh=True):
        super().__init__()
        self.video_dir = video_dir
        self.transforms = transforms
        self.include_masks = include_masks
        self.include_occ_masks = include_occ_masks
        self.color_files = sorted(glob(f"{self.video_dir}/rgb/*.jpg"))
        meta_file = self.color_files[0].replace(".jpg", ".pkl").replace("rgb", "meta")
        self.K = pickle.load(open(meta_file, "rb"))["camMat"]

        self.id_strs = []
        for i in range(len(self.color_files)):
            id = os.path.basename(self.color_files[i]).split(".")[0]
            self.id_strs.append(id)
        self.glcam_in_cvcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if do_load_mesh:
            load_res = self.get_gt_mesh()
            self.mesh = load_res["mesh"]
            self.mesh_bbox = load_res["bbox"]

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        path = self.color_files[idx]
        color = self.get_color(idx)
        depth_raw = self.get_depth(idx)
        pose = self.get_gt_pose(idx)

        if self.include_masks:
            bin_mask = self.get_mask(idx)
        else:
            bin_mask = None

        sample = get_ds_sample(
            color,
            depth_m=depth_raw,
            pose=pose,
            rgb_path=path,
            mask=bin_mask,
            intrinsics=self.K,
            transforms_rgb=self.transforms,
        )
        return sample

    def get_video_name(self):
        return os.path.dirname(os.path.abspath(self.color_files[0])).split("/")[-2]

    def get_color(self, i):
        return load_color(self.color_files[i])

    def get_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        path = f"{HO3D_ROOT}/masks_XMem/{video_name}/{index:05d}.png"
        mask = load_mask(path)
        return mask

    def get_occ_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split(".")[0])
        path = f"{HO3D_ROOT}/masks_XMem/{video_name}_hand/{index:04d}.png"
        mask = load_mask(path)
        return mask

    def get_gt_mesh(self):
        video_name = self.get_video_name()
        ob_name = None
        for k in HO3D_VIDEONAME_TO_OBJ:
            if video_name.startswith(k):
                ob_name = HO3D_VIDEONAME_TO_OBJ[k]
                break
        assert ob_name is not None, f"Could not find object name for video {video_name}"
        return load_mesh(f"{HO3D_ROOT}/models/{ob_name}/textured_simple.obj")

    def get_depth(self, i):
        depth_scale = 0.00012498664727900177
        depth = load_depth(self.color_files[i].replace(".jpg", ".png").replace("rgb", "depth"), do_convert_to_m=False)
        depth = (depth[..., 2] + depth[..., 1] * 256) * depth_scale
        return depth

    def get_gt_pose(self, i):
        meta_file = self.color_files[i].replace(".jpg", ".pkl").replace("rgb", "meta")
        meta = pickle.load(open(meta_file, "rb"))
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = self.glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt
