import os

import numpy as np
import torch


def create_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def print_args(args):
    from tabulate import tabulate

    print(tabulate(sorted(vars(args).items()), tablefmt="grid"))


def adjust_img_for_plt(img):
    img = cast_to_numpy(img)
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    if np.max(img) <= 1:
        img = img * 255
    img = img.astype(np.uint8)
    return img


def cast_to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return img
