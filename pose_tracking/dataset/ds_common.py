from collections import defaultdict

import torch


def get_ds_sample(
    color, rgb_path, depth_m=None, pose=None, mask=None, mask_visib=None, intrinsics=None, transforms=None
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
        sample["depth"] = depth
    if intrinsics is not None:
        sample["intrinsics"] = from_numpy(intrinsics)
    if mask is not None:
        if mask.max() > 1:
            mask = mask / 255.0
        sample["mask"] = from_numpy(mask)
    if pose is not None:
        sample["pose"] = from_numpy(pose)

    return sample


def from_numpy(x):
    if isinstance(x, list):
        return torch.stack([from_numpy(xx) for xx in x])
    elif isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x).float()


def process_raw_sample(sample, transforms=None):
    ds_sample = get_ds_sample(
        sample["rgb"],
        rgb_path=sample["rgb_path"],
        depth_m=sample.get("depth"),
        pose=sample.get("pose"),
        mask=sample.get("mask"),
        intrinsics=sample.get("intrinsics"),
        transforms=transforms,
    )
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
