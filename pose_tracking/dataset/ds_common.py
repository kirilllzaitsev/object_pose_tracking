from collections import defaultdict

import numpy as np
import torch
from pose_tracking.utils.rotation_conversions import convert_rotation_representation


def get_ds_sample(
    color,
    rgb_path,
    depth_m=None,
    pose=None,
    mask=None,
    mask_visib=None,
    intrinsics=None,
    transforms=None,
    priv=None,
    convert_pose_to_quat=False,
):
    if transforms is None:
        rgb = torch.from_numpy(color).permute(2, 0, 1)
    else:
        sample = transforms(image=color)
        rgb = sample["image"]

    rgb = rgb.float() / 255.0

    sample = {
        "rgb": rgb,
        "rgb_path": rgb_path,
    }
    if depth_m is not None:
        depth = from_numpy(depth_m)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        sample["depth"] = depth
    if intrinsics is not None:
        sample["intrinsics"] = from_numpy(intrinsics)
    if mask is not None:
        if mask.max() > 1:
            mask = mask / 255.0
        sample["mask"] = from_numpy(mask)
    if pose is not None:
        if convert_pose_to_quat:
            if pose.shape[-1] != 7:
                rot = torch.from_numpy(pose[:3, :3])
                quat = convert_rotation_representation(rot, rot_representation="quaternion")
                pose = np.concatenate([pose[:3, 3], quat])
        sample["pose"] = from_numpy(pose)
    if mask_visib is not None:
        sample["mask_visib"] = from_numpy(mask_visib)
    if priv is not None:
        sample["priv"] = from_numpy(priv)

    return sample


def from_numpy(x):
    if isinstance(x, list):
        return torch.stack([from_numpy(xx) for xx in x])
    elif isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x).float()


def process_raw_sample(sample, transforms=None, convert_pose_to_quat=False):
    ds_sample = get_ds_sample(
        sample["rgb"],
        rgb_path=sample["rgb_path"],
        depth_m=sample.get("depth"),
        pose=sample.get("pose"),
        mask=sample.get("mask"),
        intrinsics=sample.get("intrinsics"),
        transforms=transforms,
        convert_pose_to_quat=convert_pose_to_quat,
        priv=sample.get("priv"),
    )
    # add keys present in sample but not in ds_sample
    for k, v in sample.items():
        if k not in ds_sample:
            ds_sample[k] = v
    return ds_sample


def dict_collate_fn(batch):
    # result is a dict with values of size batch_size x ...
    new_b = defaultdict(list)
    for k in batch[0].keys():
        new_b[k] = [d[k] for d in batch]
    for k, v in new_b.items():
        if isinstance(v[0], torch.Tensor):
            new_b[k] = torch.stack(v)
    return new_b


def seq_collate_fn(batch):
    # result is a list of size seq_len with dicts having values of size batch_size x ...
    new_b = []
    for i in range(len(batch[0])):
        new_b.append(dict_collate_fn([d[i] for d in batch]))
    return new_b
