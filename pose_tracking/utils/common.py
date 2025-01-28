import glob
import os
import re

import numpy as np
import torch


def create_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def print_args(args, logger=None):
    from tabulate import tabulate

    msg = tabulate(sorted(vars(args).items()), tablefmt="grid")
    if logger:
        logger.info(msg)
    else:
        print(msg)


def adjust_img_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if np.max(img) <= 1:
        img = img * 255
    img = img.astype(np.uint8)
    return img


def adjust_depth_for_plt(img):
    img = cast_to_numpy(img)
    if len(img.shape) == 4:
        if img.shape[0] == 1:
            img = img[0]
        else:
            raise RuntimeError(f"Expected 1 image, got {img.shape[0]}")
    if img.shape[0] == 1:
        img = img.transpose(1, 2, 0)
    return img


def cast_to_numpy(x, dtype=None) -> np.ndarray:
    if x is None or isinstance(x, str):
        return x
    elif isinstance(x, list):
        return np.array([cast_to_numpy(xx) for xx in x])
    elif isinstance(x, dict):
        return {k: cast_to_numpy(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float, complex)):
        return x
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def cast_to_torch(x, device=None, include_top_list=False):
    if x is None or isinstance(x, str):
        return x
    elif type(x) in [list, tuple]:
        res = [cast_to_torch(xx, device=device) for xx in x]
        if include_top_list:
            return torch.stack(res)
        return res
    elif isinstance(x, dict):
        return {k: cast_to_torch(v, device=device) for k, v in x.items()}
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float, complex)):
        return torch.tensor(x, device=device)
    return x.to(device)


def detach_and_cpu(x):
    if isinstance(x, list):
        return [detach_and_cpu(xx) for xx in x]
    elif isinstance(x, dict):
        return {k: detach_and_cpu(v) for k, v in x.items()}
    elif isinstance(x, np.ndarray) or isinstance(x, (int, float, complex)) or np.isscalar(x):
        return x
    elif x is None:
        return x
    return x.detach().cpu()


def convert_arr_to_tensor(v):
    if isinstance(v[0], np.ndarray):
        v = [torch.from_numpy(x) for x in v]
    v_tensor = torch.stack(v)
    return v_tensor


def extract_idxs(arr, idxs, do_extract_dict_contents=False):
    if arr is None:
        return None
    elif isinstance(arr, list):
        if len(arr) == 0:
            return []
        if do_extract_dict_contents and len(arr) > 0 and isinstance(arr[0], dict):
            return [extract_idxs(a, idxs, do_extract_dict_contents=do_extract_dict_contents) for a in arr]
        else:
            return [arr[idx] for idx in idxs]
    elif isinstance(arr, dict):
        return {k: extract_idxs(v, idxs, do_extract_dict_contents=do_extract_dict_contents) for k, v in arr.items()}
    return arr[idxs]


def istensor(x):
    return isinstance(x, torch.Tensor)


def get_ordered_paths(pattern, sort_fn=None):
    sort_fn = sort_fn or (lambda x: int(re.search(r".*?(\d+)(?!.*\d)", x).group(1)))  # search for last numerical value
    pattern = str(pattern)
    if "*" not in pattern:
        assert os.path.isdir(pattern), f"Check {pattern=}"
        pattern = f"{pattern}/*"
    return sorted(glob.glob(pattern), key=sort_fn)
