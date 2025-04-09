"""
.
├── cam_K.txt
├── depth
│   └── *.png
├── masks
│   └── *.png
├── mesh
│   ├── *.obj
│   ├── *.obj.mtl
│   ├── images
│   │   └── *.pdf
│   └── textures
│       └── *.png
└── rgb
    └── *.png
"""

import os

import numpy as np
from pose_tracking.dataset.tracking_ds import TrackingDataset


class CustomDataset(TrackingDataset):

    ds_name = "custom"

    def __init__(
        self,
        *args,
        mesh_path=None,
        obj_name="cube",
        do_convert_depth_to_m=True,
        pose_dirname="poses",
        **kwargs,
    ):
        super().__init__(*args, obj_name=obj_name, do_convert_depth_to_m=do_convert_depth_to_m, pose_dirname=pose_dirname, **kwargs)
        if mesh_path is None:
            mesh_path_prop = f"{self.video_dir}/mesh/cube.obj"
            if os.path.exists(mesh_path_prop):
                mesh_path = mesh_path_prop
        if mesh_path is not None:
            self.set_up_obj_mesh(mesh_path)

    def augment_sample(self, sample, idx):
        sample["class_id"] = [0]
        return sample


class CustomDatasetTest(CustomDataset):

    ds_name = "custom_test"

    def __init__(self, *args, pose_dirname="pose", **kwargs):
        if not os.path.exists(f"{kwargs['video_dir']}/pose"):
            fpose_save_dirname = 'ob_in_cam_fpose'
            if os.path.exists(f"{kwargs['video_dir']}/{fpose_save_dirname}"):
                kwargs["include_pose"] = True
                kwargs["pose_dirname"] = fpose_save_dirname
            else:
                kwargs["include_pose"] = False
                kwargs["include_bbox_2d"] = False
        mask_dir = f"{kwargs['video_dir']}/masks"
        kwargs["include_mask"] = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
        # kwargs["target_hw"] = (480, 640)
        super().__init__(*args, pose_dirname=pose_dirname, **kwargs)

    def augment_sample(self, sample, idx):
        sample = super().augment_sample(sample, idx)
        if not self.include_pose:
            sample["pose"] = np.eye(4)
        return sample
