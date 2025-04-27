import copy

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
from pose_tracking.utils.segm_utils import infer_bounding_box
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
        ts.append(A.MotionBlur(p=0.4, blur_limit=(3, 21)))
    if "gamma" in transform_names:
        ts.append(A.RandomGamma(p=0.1, gamma_limit=(60, 140)))
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


def mask_pixels_torch(img: torch.Tensor, p=0.3, pixels_masked_max_percent=0.15, use_noise=False, use_blocks=True):
    """
    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W).
        p (float): Probability of applying the masking.
        pixels_masked_max_percent (float): Max fraction of total pixels to modify.

    Returns:
        torch.Tensor: Modified image.
    """
    if img.ndim == 4:
        return torch.stack(
            [mask_pixels_torch(img[i], p, pixels_masked_max_percent) for i in range(img.shape[0])], dim=0
        )
    if torch.rand(1).item() > p:
        return img

    mask = torch.rand_like(img) > torch.rand(1).item() * pixels_masked_max_percent
    res = img * mask
    if use_noise:
        noise = torch.randn_like(img) * 0.01
        res += noise * ~mask
    if use_blocks:
        res = mask_random_block(res, max_num_blocks=20, max_block_size=32)
    return res


def mask_random_block(img, max_num_blocks=20, max_block_size=32):
    if img.ndim == 4:
        return torch.stack([mask_random_block(img[i], max_num_blocks, max_block_size) for i in range(img.shape[0])])
    B, H, W = img.shape
    num_blocks = random.randint(0, max_num_blocks)
    for _ in range(num_blocks):
        block_size = random.randint(1, max_block_size)
        y = random.randint(0, H - block_size)
        x = random.randint(0, W - block_size)
        img[:, y : y + block_size, x : x + block_size] = 0.0
    return img


def noisify_pose(pose, angle_mean=0, angle_std=3, t_mean=0, t_std=0.01):
    delta_pose_noise = copy.deepcopy(pose)
    angles = np.random.normal(angle_mean, angle_std, 3)
    rot_noise = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    t_noise = np.random.normal(t_mean, t_std, 3)
    delta_pose_noise[:3, 3] += t_noise
    delta_pose_noise[:3, :3] = delta_pose_noise[:3, :3] @ rot_noise
    return delta_pose_noise


def generate_random_mask_on_obj(obj_mask, bbox=None, num_vertices=5):
    from pose_tracking.utils.vis import convert_bbox_to_min_max_corners

    if bbox is None:
        bbox = infer_bounding_box(obj_mask)
    bbox_xy_ul, bbox_xy_br = convert_bbox_to_min_max_corners(bbox)

    h, w = obj_mask.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    shape_type = np.random.choice(["polygon", "circle", "ellipse", "rectangle"])
    shape_type = "ellipse"

    def sample_pts(num_vertices):
        x_candidates = torch.arange(bbox_xy_ul[0], bbox_xy_br[0] + 1)
        y_candidates = torch.arange(bbox_xy_ul[1], bbox_xy_br[1] + 1)

        center_x = (bbox_xy_ul[0] + bbox_xy_br[0]) / 2.0
        center_y = (bbox_xy_ul[1] + bbox_xy_br[1]) / 2.0

        sigma_x = (bbox_xy_br[0] - bbox_xy_ul[0]) / 3.0
        sigma_y = (bbox_xy_br[1] - bbox_xy_ul[1]) / 3.0

        weights_x = torch.exp(-0.5 * ((x_candidates - center_x) / sigma_x) ** 2)
        weights_y = torch.exp(-0.5 * ((y_candidates - center_y) / sigma_y) ** 2)

        weights_x /= weights_x.sum()
        weights_y /= weights_y.sum()
        weights_x = weights_x.numpy()
        weights_y = weights_y.numpy()

        pts = []
        for _ in range(num_vertices):
            x = np.random.choice(x_candidates, p=weights_x)
            y = np.random.choice(y_candidates, p=weights_y)
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        return pts

    if shape_type == "polygon":
        pts = sample_pts(num_vertices)[:, None]
        cv2.fillPoly(mask, [pts], 0)

    elif shape_type == "circle":
        cx = np.random.randint(bbox_xy_ul[0], bbox_xy_br[0])
        cy = np.random.randint(bbox_xy_ul[1], bbox_xy_br[1])
        max_radius = min(bbox_xy_br[0] - bbox_xy_ul[0], bbox_xy_br[1] - bbox_xy_ul[1]) // 3
        radius = np.random.randint(max_radius // 2, max_radius)
        cv2.circle(mask, (cx, cy), radius, 0, thickness=-1)

    elif shape_type == "ellipse":
        cx = np.random.randint(bbox_xy_ul[0], bbox_xy_br[0])
        cy = np.random.randint(bbox_xy_ul[1], bbox_xy_br[1])
        axis1 = np.random.randint((bbox_xy_br[0] - bbox_xy_ul[0]) // 5, (bbox_xy_br[0] - bbox_xy_ul[0]) // 3)
        axis2 = np.random.randint((bbox_xy_br[1] - bbox_xy_ul[1]) // 5, (bbox_xy_br[1] - bbox_xy_ul[1]) // 3)
        angle = np.random.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), (axis1, axis2), angle, 0, 360, 0, thickness=-1)

    elif shape_type == "rectangle":
        pts = sample_pts(2)
        x_ur, y_ur = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_lr, y_lr = np.max(pts[:, 0]), np.max(pts[:, 1])
        cv2.rectangle(mask, (x_ur, y_ur), (x_lr, y_lr), 0, thickness=-1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask > 0


class RandomPolygonMask:
    def __init__(self, num_vertices=5, always_apply=False, p=0.5):
        """
        num_vertices: number of vertices of the polygon mask.
        p: probability of applying the transform.
        """
        self.num_vertices = num_vertices
        self.always_apply = always_apply
        self.p = p

    def apply(self, img, **params):
        if np.random.rand() > self.p:
            return img
        obj_mask = params["mask"]
        bbox = infer_bounding_box(obj_mask)
        if bbox is not None:
            bbox_xy_ul, bbox_xy_br = bbox
            if bbox_xy_br[0] <= bbox_xy_ul[0] or bbox_xy_br[1] <= bbox_xy_ul[1]:
                return img
        try:
            mask = generate_random_mask_on_obj(obj_mask, bbox=bbox, num_vertices=self.num_vertices)
        except Exception as e:
            print(f"Error generating mask: {e}")
            return img
        if img.ndim == 3:
            obj_mask = obj_mask[..., None]
            mask = mask[..., None]
        fill_value = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        img = np.where(np.bitwise_and(obj_mask > 0, mask == 0), fill_value, img)
        return img

    def get_transform_init_args_names(self):
        return "num_vertices"
