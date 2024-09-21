import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from bop_toolkit_lib.dataset.bop_imagewise import (
    io_load_masks,
    load_bop_depth,
    load_bop_rgb,
)
from PIL import Image
from pose_tracking.dataset.bop_loaders import load_cad, load_list_scene, load_metadata
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
        seq_length=1,
        step_skip=1,
        seq_start=0,
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
            if cad_dir is None:
                possible_cad_subdir_names = ["models_cad", "models"]
                for subdir_name in possible_cad_subdir_names:
                    cad_dir = root_dir / subdir_name
                    if cad_dir.exists():
                        break
                assert cad_dir is not None, "CAD dir must be provided"
            self.cads = load_cad(cad_dir)
            self.cad_dir = cad_dir
        else:
            self.cads = None

        self.trajs = []
        self.seq_length = seq_length
        self.step_skip = step_skip
        self.seq_start = seq_start
        for scene_id, traj_objs in list(self.metadata.groupby("scene_id")):
            traj_objs_grouped = traj_objs.groupby("frame_id")
            self.trajs.append((scene_id, traj_objs))

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        seq_start = self.seq_start
        scene_id, traj_objs = self.trajs[idx]

        def parse_tensor_from_str(x, shape=(4, 4)):
            return torch.tensor(np.array(np.matrix(x, dtype=np.float32)).reshape(shape))

        traj_objs.pose = traj_objs.pose.apply(lambda x: parse_tensor_from_str(x, shape=(4, 4)))
        traj_objs.intrinsic = traj_objs.intrinsic.apply(lambda x: parse_tensor_from_str(x, shape=(3, 3)))
        # discard frame_ids
        traj = [to[1] for to in traj_objs.groupby("frame_id")]
        if seq_start is None:
            seq_start = torch.randint(
                0,
                max(1, len(traj) + 1 - self.seq_length * self.step_skip),
                (1,),
            ).item()
        assert self.step_skip > 0, f"{self.step_skip=}"
        sample_traj_seq = defaultdict(list)

        for ts in range(self.seq_length):
            frame_idx = seq_start + ts * self.step_skip
            sample = traj[frame_idx].to_dict(orient="list")
            masks = np.array([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in sample["mask_path"]])
            masks_visib = np.array([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in sample["mask_visib_path"]])
            depth = load_bop_depth(sample["depth_path"][0])
            rgb = load_bop_rgb(sample["rgb_path"][0])

            sample["mask"] = masks
            sample["mask_visib"] = masks_visib
            sample["depth"] = depth
            sample["rgb"] = rgb

            for k in [
                "scene_id",
                "frame_id",
                "obj_id",
                "idx_obj",
                "pose",
                "rgb_path",
                "mask_path",
                "mask_visib_path",
                "depth_path",
                "visib_fract",
                "bbox_obj",
                "bbox_visib",
                "intrinsic",
                "rgb",
                "depth",
                "mask",
                "mask_visib",
            ]:
                sample_traj_seq[k].append(sample[k])

        for k in ["rgb", "depth", "mask", "mask_visib"]:
            v = sample_traj_seq[k]
            if isinstance(v[0], np.ndarray):
                v = [torch.from_numpy(x) for x in v]
            sample_traj_seq[k] = torch.stack(v)

        return sample_traj_seq


if __name__ == "__main__":
    from pose_tracking.config import DATA_DIR

    for dataset_name, split_ in zip(["tless/test", "tless/train"], ["test_primesense", "train_primesense"]):
        ds_dir = DATA_DIR / dataset_name
        dataset = BaseBOP(ds_dir, split_, cad_dir=DATA_DIR / "tless/models_cad", do_load_cad=True)

    dataset_names = ["lmo"]
    for dataset_name in tqdm(dataset_names):
        ds_dir = DATA_DIR / dataset_name
        splits = [s for s in os.listdir(ds_dir) if os.path.isdir(ds_dir / s)]
        splits = [s for s in splits if s.startswith("train") or s.startswith("val") or s.startswith("test")]
        for split_ in splits:
            dataset = BaseBOP(ds_dir, split_)
