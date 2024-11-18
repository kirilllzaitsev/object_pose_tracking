import albumentations as A
import numpy as np
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(use_norm=False):
    ts = []
    ts.append(ToTensorV2())
    if use_norm:
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ts.append(norm)
    return Compose(ts)


def mask_pixels(img, p=0.5, pixels_masked_max_percent=0.1):
    if len(img.shape) == 2:
        us, vs = np.where(img > 0)
        ch_axis = None
    else:
        if img.shape[0] in [1, 3]:
            ch_axis = 0
        else:
            ch_axis = 2
        us, vs = np.where(np.any(img > 0, axis=ch_axis))
    if np.random.uniform() < p:
        pixels_masked_share = np.random.uniform(0, pixels_masked_max_percent)
        pxs = np.random.choice(np.arange(0, len(us)), int(pixels_masked_share * len(us)), replace=False)
        if ch_axis is None:
            img[us[pxs], vs[pxs]] = 0
        elif ch_axis == 0:
            img[:, us[pxs], vs[pxs]] = 0
        else:
            img[us[pxs], vs[pxs], :] = 0
    return img
