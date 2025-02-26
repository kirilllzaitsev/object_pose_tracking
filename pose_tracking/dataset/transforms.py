import copy

import albumentations as A
import numpy as np
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from scipy.spatial.transform import Rotation


def get_transforms(transform_names=None, transform_prob=0.75):
    transform_names = transform_names or []
    ts = []
    if "jitter" in transform_names:
        ts.append(A.ColorJitter(p=0.5))
    if "iso" in transform_names:
        ts.append(A.ISONoise(p=0.25))
    if "brightness" in transform_names:
        ts.append(A.RandomBrightnessContrast(p=0.75, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2)))
    if "blur" in transform_names:
        ts.append(A.Blur(p=0.1, blur_limit=(3, 5)))
    if "motion_blur" in transform_names:
        ts.append(A.MotionBlur(p=0.2, blur_limit=(3, 11)))
    if "gamma" in transform_names:
        ts.append(A.RandomGamma(p=0.1, gamma_limit=(80, 120)))
    if "hue" in transform_names:
        ts.append(A.HueSaturationValue(p=0.1, val_shift_limit=(-40, -20)))
    if "norm" in transform_names:
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ts.append(norm)
    rgb_transforms = Compose(ts, p=transform_prob)
    return Compose([rgb_transforms, ToTensorV2()])


def get_transforms_video(transforms):
    return A.ReplayCompose(transforms.transforms)


def apply_replay_transform(rgb, transform_res):
    return A.ReplayCompose.replay(transform_res["replay"], image=rgb)


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


def noisify_pose(pose, angle_mean=0, angle_std=3, t_mean=0, t_std=0.01):
    delta_pose_noise = copy.deepcopy(pose)
    angles = np.random.normal(angle_mean, angle_std, 3)
    rot_noise = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    t_noise = np.random.normal(t_mean, t_std, 3)
    delta_pose_noise[:3, 3] += t_noise
    delta_pose_noise[:3, :3] = delta_pose_noise[:3, :3] @ rot_noise
    return delta_pose_noise
