from collections import defaultdict

import numpy as np
import torch
from pose_tracking.dataset.transforms import mask_pixels
from pose_tracking.utils.common import cast_to_numpy, cast_to_torch
from pose_tracking.utils.geom import convert_3d_t_for_2d
from pose_tracking.utils.rotation_conversions import convert_rotation_representation


def get_ds_sample(
    color,
    rgb_path,
    depth_m=None,
    pose=None,
    mask=None,
    mask_visib=None,
    intrinsics=None,
    transforms_rgb=None,
    priv=None,
    do_convert_pose_to_quat=False,
    mask_pixels_prob=0.0,
    pixels_masked_max_percent=0.1,
    rot_repr="quaternion",
    t_repr="3d",
):
    if transforms_rgb is not None:
        transformed = transforms_rgb(image=color)
        color = transformed["image"]

    rgb = adjust_img_for_torch(color)

    sample = {
        "rgb": rgb,
        "rgb_path": rgb_path,
    }
    if depth_m is not None:
        depth = from_numpy(depth_m)
        if mask_pixels_prob > 0:
            depth = mask_pixels(depth, p=mask_pixels_prob, pixels_masked_max_percent=pixels_masked_max_percent)
        depth = adjust_depth_for_torch(depth)
        sample["depth"] = depth
    if intrinsics is not None:
        sample["intrinsics"] = from_numpy(intrinsics)
    if mask is not None:
        mask = adjust_mask_for_torch(mask)
        sample["mask"] = from_numpy(mask)
    if pose is not None:
        rot = pose[:3, :3]
        t = pose[:3, 3]
        if t_repr == "2d":
            hw = rgb.shape[-2:]
            t_2d_norm, center_depth = convert_3d_t_for_2d(cast_to_torch(t), cast_to_torch(intrinsics), hw)
            sample["center_depth"] = center_depth[None]
            sample["xy"] = t_2d_norm
        if do_convert_pose_to_quat and rot_repr != "None":
            quat = convert_rotation_representation(torch.from_numpy(rot), rot_representation=rot_repr)
            pose = np.concatenate([t, quat])

        sample["pose"] = from_numpy(pose)
    if mask_visib is not None:
        sample["mask_visib"] = from_numpy(mask_visib)
    if priv is not None:
        sample["priv"] = from_numpy(priv)

    return sample


def adjust_depth_for_torch(depth):
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    return depth


def adjust_mask_for_torch(mask):
    if mask.max() > 1:
        mask = mask / 255.0
    return mask


def adjust_img_for_torch(rgb):
    if isinstance(rgb, np.ndarray):
        rgb = from_numpy(rgb)
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(2, 0, 1)
    rgb = rgb.float()
    if rgb.max() > 1:
        rgb /= 255.0
    return rgb


def from_numpy(x):
    if isinstance(x, list):
        return torch.stack([from_numpy(xx) for xx in x])
    elif isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x).float()


def process_raw_sample(sample, *args, **kwargs):
    ds_sample = get_ds_sample(
        sample["rgb"],
        rgb_path=sample["rgb_path"],
        depth_m=sample.get("depth"),
        pose=sample.get("pose"),
        mask=sample.get("mask"),
        intrinsics=sample.get("intrinsics"),
        priv=sample.get("priv"),
        mask_visib=sample.get("mask_visib"),
        *args,
        **kwargs,
    )
    # add keys present in sample but not in ds_sample
    for k, v in sample.items():
        if k not in ds_sample:
            if v is not None and not ("path" in k or "name" in k) and not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            ds_sample[k] = v
    return ds_sample


def dict_collate_fn(batch):
    # result is a dict with values of size batch_size x ...
    new_b = defaultdict(list)
    for k in batch[0].keys():
        new_b[k] = [d[k] for d in batch]
    for k, v in new_b.items():
        if k in ["class_id", "bbox_2d", "xyz_map"]:
            continue
        if isinstance(v[0], torch.Tensor):
            new_b[k] = torch.stack(v)
    return new_b


def seq_collate_fn(batch):
    # result is a list of size seq_len with dicts having values of size batch_size x ...
    new_b = []
    seq_lens = [len(d) for d in batch]
    for i in range(min(seq_lens)):
        new_b.append(dict_collate_fn([d[i] for d in batch]))
    return new_b


def batch_seq_collate_fn(batch):
    # result is a tensor of size batch_size with dicts having values of size seq_len x ...
    new_b = []
    for i in range(len(batch)):
        new_b.append(dict_collate_fn(batch[i]))
    new_b = dict_collate_fn(new_b)
    return new_b


def convert_seq_batch_to_batch_seq(batch, keys=None):
    # from keyxseq_lenxbatch to batchxseq_lenxkey
    res = []
    keys = keys or batch.keys()
    for bidx in range(len(batch["rgb"][0])):
        news = []
        for sidx in range(len(batch["rgb"])):
            news.append({k: v[sidx][bidx] for k, v in batch.items() if len(v) > 0 if k in keys})
        res.append(news)
    return res


def convert_batch_seq_to_seq_batch(batch, keys=None):
    # from keyxbatchxseq_len to seq_lenxkeyxbatch
    res = []
    keys = keys or batch.keys()
    for sidx in range(len(batch["rgb"][0])):
        news = {}
        for k, v in batch.items():
            if k in keys:
                news[k] = [v[bidx][sidx] for bidx in range(len(v))]
                news[k] = cast_to_numpy(news[k])
        res.append(news)
    return res
