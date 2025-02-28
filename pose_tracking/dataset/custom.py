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

from pose_tracking.dataset.tracking_ds import TrackingDataset


class CustomDataset(TrackingDataset):

    ds_name = "custom"

    def __init__(
        self,
        *args,
        mesh_path=None,
        **kwargs,
    ):
        kwargs["do_convert_depth_to_m"] = True
        kwargs["pose_dirname"] = "poses"
        super().__init__(*args, **kwargs)
        if mesh_path is None:
            mesh_path_prop = f"{self.video_dir}/mesh/cube.obj"
            if os.path.exists(mesh_path_prop):
                mesh_path = mesh_path_prop
        if mesh_path is not None:
            self.set_up_obj_mesh(mesh_path)

    def augment_sample(self, sample, idx):
        sample["class_id"] = [0]
        return sample
