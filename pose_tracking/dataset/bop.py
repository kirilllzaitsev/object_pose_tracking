import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pose_tracking import logger
from pose_tracking.dataset.bop_loaders import load_cad, load_list_scene, load_metadata
from pose_tracking.utils.io import load_json
from pose_tracking.utils.pose import combine_R_and_T
from pose_tracking.utils.trimesh_utils import load_mesh
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseBOP(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        rot_repr="rotation6d",
        cad_dir=None,
        do_load_cad=False,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split
        self.rot_repr = rot_repr
        self.cad_dir = cad_dir
        self.list_scenes = load_list_scene(root_dir, split)
        self.metadata = load_metadata(root_dir, split)
        if do_load_cad:
            assert cad_dir is not None, "CAD dir must be provided"
            self.cads = load_cad(cad_dir)
        else:
            self.cads = None


if __name__ == "__main__":
    from pose_tracking.config import PROJ_DIR

    dataset_names = ["lmo"]
    # tless is special
    for dataset_name, split_ in zip(["tless/test", "tless/train"], ["test_primesense", "train_primesense"]):
        ds_dir = PROJ_DIR / "data" / dataset_name
        dataset = BaseBOP(ds_dir, split_, cad_dir=PROJ_DIR / "data" / "tless/models_cad", do_load_cad=True)
        for scene_path in dataset.list_scenes:
            scene_id = scene_path.split("/")[-1]
            ...

    for dataset_name in tqdm(dataset_names):
        ds_dir = PROJ_DIR / "data" / dataset_name
        splits = [s for s in os.listdir(ds_dir) if os.path.isdir(ds_dir / s)]
        splits = [s for s in splits if s.startswith("train") or s.startswith("val") or s.startswith("test")]
        for split_ in splits:
            dataset = BaseBOP(ds_dir, split_)
            ...
