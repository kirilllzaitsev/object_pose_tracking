import os

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
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if np.max(img) <= 1:
        img = img * 255
    img = img.astype(np.uint8)
    return img


def cast_to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    else:
        if isinstance(arr[0], torch.Tensor):
            arr = [cast_to_numpy(a) for a in arr]
    return arr


def convert_arr_to_tensor(v):
    if isinstance(v[0], np.ndarray):
        v = [torch.from_numpy(x) for x in v]
    v_tensor = torch.stack(v)
    return v_tensor
