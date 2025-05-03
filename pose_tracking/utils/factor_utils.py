import cv2
import numpy as np
import torch
from pose_tracking.dataset.ds_common import adjust_img_for_torch
from pose_tracking.utils.common import cast_to_numpy, cast_to_torch
from torchvision.transforms.functional import rgb_to_grayscale


def get_visib_px_num(mask):
    return mask.sum()


def calc_factor_strength(x_num, min_val, max_val, p=2):
    # res = 1 - ((x_num - min_val) / (max_val - min_val)) ** p
    # res = 1 - 1 / (1 + np.exp(-0.01 * (x_num - (min_val + max_val) / 2)))
    res = (x_num - min_val) / (max_val - min_val)
    return res


def calc_texture_factor(rgb: torch.Tensor, mask: torch.Tensor, bins: int = 32, sigma: float = 2.0) -> float:
    """smoothed histogram+entropy"""
    if mask.ndim == 2:
        mask = mask[None]
    mask = cast_to_torch(mask)
    rgb = adjust_img_for_torch(rgb)
    gray = rgb_to_grayscale(rgb)
    pixels = gray[mask.bool()]
    if pixels.numel() == 0:
        return 0.0

    bin_edges = torch.linspace(0, 1, bins, device=gray.device)
    bin_width = bin_edges[1] - bin_edges[0]

    diffs = pixels[:, None] - bin_edges[None, :]
    weights = torch.exp(-0.5 * (diffs / (sigma * bin_width)) ** 2)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

    hist = weights.sum(dim=0)
    prob = hist / hist.sum()

    eps = 1e-9
    entropy = -(prob * (prob + eps).log2()).sum()
    return entropy.item()


def calc_bbox_area(bbox):
    # ul,br
    x_min, y_min = bbox[0]
    x_max, y_max = bbox[1]
    return max(1, (x_max - x_min)) * max(1, (y_max - y_min))


def rasterize_bbox_cv(points_2d, img_size: tuple) -> np.ndarray:
    hull = cv2.convexHull(cast_to_numpy(points_2d).astype(np.int32))
    mask = np.zeros(img_size, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)
