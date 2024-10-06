import torch


def get_ds_sample(color, depth_m, rgb_path, pose=None, mask=None, intrinsics=None, transforms=None):
    if transforms is None:
        rgb = torch.from_numpy(color).permute(2, 0, 1)
    else:
        sample = transforms(image=color)
        rgb = sample["image"]

    rgb = rgb.float() / 255.0

    depth = torch.from_numpy(depth_m).float()

    sample = {
        "rgb": rgb,
        "depth": depth,
        "rgb_path": rgb_path,
    }
    if intrinsics is not None:
        sample["intrinsics"] = torch.from_numpy(intrinsics).float()
    if mask is not None:
        sample["mask"] = torch.from_numpy(mask).float()
    if pose is not None:
        sample["pose"] = torch.from_numpy(pose).float()

    return sample


def process_raw_sample(sample, transforms=None):
    return get_ds_sample(
        sample["rgb"],
        sample["depth"],
        rgb_path=sample["rgb_path"],
        pose=sample.get("pose"),
        mask=sample.get("mask"),
        intrinsics=sample.get("intrinsics"),
        transforms=transforms,
    )
