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
        self.set_up_obj_mesh(mesh_path)
