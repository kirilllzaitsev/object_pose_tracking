import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pose_tracking.config import DATA_DIR
from pose_tracking.dataset.bop_loaders import load_cad, load_list_scene, load_metadata
from pose_tracking.utils.common import convert_arr_to_tensor
from pose_tracking.utils.io import load_depth, load_mask, load_pose, load_color
from torch.utils.data import Dataset
from tqdm import tqdm


class BOPDataset(Dataset):
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
        use_keyframes=False,
        keyframe_path=None,
        include_rgb=True,
        include_masks=False,
        include_depth=False,
        depth_scaler_to_mm=1.0,
    ):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.rot_repr = rot_repr
        self.cad_dir = cad_dir
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.include_masks = include_masks
        self.depth_scaler_to_mm = depth_scaler_to_mm
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

        if use_keyframes:
            if keyframe_path is None:
                keyframe_path = self.root_dir / "keyframe.txt"
                assert keyframe_path.exists(), f"{keyframe_path} does not exist"
            self.keyframes = load_keyframes(keyframe_path)

        self.trajs = []
        self.seq_length = seq_length
        self.step_skip = step_skip
        self.seq_start = seq_start
        for scene_id, frame_obj_flat in list(self.metadata.groupby("scene_id")):
            if use_keyframes:
                scene_id_format = format_scene_id(scene_id)
                if scene_id_format not in self.keyframes:
                    continue
                frame_obj_flat = frame_obj_flat[
                    frame_obj_flat["frame_id"].apply(lambda x: format_frame_id(x) in self.keyframes[scene_id_format])
                ]
            self.trajs.append((scene_id, frame_obj_flat))

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        seq_start = self.seq_start
        scene_id, traj_objs = self.trajs[idx]

        # discard frame_ids
        traj = [to for (_, to) in traj_objs.groupby("frame_id")]
        timesteps = self.seq_length if self.seq_length is not None else len(traj)
        if seq_start is None:
            seq_start = torch.randint(
                0,
                max(1, len(traj) + 1 - timesteps * self.step_skip),
                (1,),
            ).item()
        assert self.step_skip > 0, f"{self.step_skip=}"
        sample_traj_seq = defaultdict(list)

        for ts in range(timesteps):
            frame_idx = seq_start + ts * self.step_skip
            sample = traj[frame_idx].to_dict(orient="list")
            sample["pose"] = [parse_tensor_from_str(x, shape=(4, 4)) for x in sample["pose"]]
            sample["intrinsics"] = [parse_tensor_from_str(x, shape=(3, 3)) for x in sample["intrinsic"]]

            if self.include_rgb:
                rgb = load_color(sample["rgb_path"][0])
                rgb = rgb.astype(np.float32)
                rgb /= 255.0
                rgb = torch.from_numpy(rgb).permute(2, 0, 1)
                sample["rgb"] = rgb

            if self.include_depth:
                depth = load_depth(sample["depth_path"][0])
                sample["depth"] = (depth * self.depth_scaler_to_mm) / 1000
                sample_traj_seq["depth"].append(sample["depth"])

            if self.include_masks:
                masks = np.array([load_mask(p) for p in sample["mask_path"]])
                masks_visib = np.array([load_mask(p) for p in sample["mask_visib_path"]])
                sample["mask"] = masks.astype(float) / 255.0
                sample["mask_visib"] = masks_visib.astype(float) / 255.0

            keys_in_sample = [
                "obj_id",
                "idx_obj",
                "pose",
                "visib_fract",
                "bbox_obj",
                "bbox_visib",
                "mask_path",
                "mask_visib_path",
            ]
            if self.include_rgb:
                keys_in_sample.append("rgb")
            if self.include_masks:
                keys_in_sample.extend(["mask", "mask_visib"])

            for k in keys_in_sample:
                sample_traj_seq[k].append(sample[k])
            for k in [
                "rgb_path",
                "depth_path",
                "intrinsics",
            ]:
                sample_traj_seq[k].append(sample[k][0])

            sample_traj_seq["scene_id"].append(format_scene_id(sample["scene_id"][0]))
            sample_traj_seq["frame_id"].append(format_frame_id(sample["frame_id"][0]))

        keys_to_convert = []
        if self.include_rgb:
            keys_to_convert.append("rgb")
        if self.include_masks:
            keys_to_convert.extend(["mask", "mask_visib"])
        if self.include_depth:
            keys_to_convert.append("depth")
        for k in keys_to_convert:
            v_tensor = convert_arr_to_tensor(sample_traj_seq[k])
            sample_traj_seq[k] = v_tensor

        return sample_traj_seq


class BOPDatasetBenchmark(BOPDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["include_rgb"] = False
        kwargs["include_depth"] = False
        kwargs["include_masks"] = False
        super().__init__(*args, **kwargs)


class BOPDatasetEval(BOPDataset):
    def __init__(self, preds_path, *args, **kwargs):
        kwargs["include_rgb"] = True
        kwargs["include_depth"] = True
        kwargs["include_masks"] = True
        super().__init__(*args, **kwargs)
        self.preds_path = Path(preds_path)

    def get_pred_pose(self, filename_no_ext):
        pose = load_pose(self.preds_path / f"{filename_no_ext}.txt")
        return pose

    def __getitem__(self, i):
        sample = super().__getitem__(i)
        filename_no_ext = [f"{Path(p).stem}" for p in sample["rgb_path"]]
        sample["pose_pred"] = convert_arr_to_tensor([self.get_pred_pose(f) for f in filename_no_ext])
        return sample


def load_keyframes(path):
    keyframes = open(path).readlines()
    keyframes_grouped = defaultdict(list)

    for idx, k in enumerate(keyframes):
        scene_id, frame_id = k.strip().split("/")
        new_scene_id = format_scene_id(scene_id)
        new_frame_id = format_frame_id(frame_id)
        keyframes_grouped[new_scene_id].append(new_frame_id)
    return keyframes_grouped


def format_scene_id(scene_id):
    return f"{int(scene_id):06d}"


def format_frame_id(scene_id):
    return f"{int(scene_id):06d}"


def parse_tensor_from_str(x, shape=(4, 4)):
    return torch.tensor(np.array(np.matrix(x, dtype=np.float32)).reshape(shape))


def check_data():
    for dataset_name, split_ in zip(["tless/test", "tless/train"], ["test_primesense", "train_primesense"]):
        ds_dir = DATA_DIR / dataset_name
        dataset = BOPDataset(ds_dir, split_, cad_dir=DATA_DIR / "tless/models_cad", do_load_cad=True)

    dataset_names = ["lmo"]
    for dataset_name in tqdm(dataset_names):
        ds_dir = DATA_DIR / dataset_name
        splits = [s for s in os.listdir(ds_dir) if os.path.isdir(ds_dir / s)]
        splits = [s for s in splits if s.startswith("train") or s.startswith("val") or s.startswith("test")]
        for split_ in splits:
            dataset = BOPDataset(ds_dir, split_)
