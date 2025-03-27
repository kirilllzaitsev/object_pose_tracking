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

    def __init__(self, *args, **kwargs):
        kwargs["include_pose"] = False
        kwargs["include_depth"] = False
        kwargs["include_mask"] = False
        kwargs["include_bbox_2d"] = False
        kwargs["target_size"] = (480, 640)
        super().__init__(*args, **kwargs)

    def augment_sample(self, sample, idx):
        sample = super().augment_sample(sample, idx)
        sample["pose"] = np.eye(4)
        return sample
