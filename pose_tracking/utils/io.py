import json
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as cocomask


def parse_rle_mask(seg_obj):
    h, w = seg_obj["size"]
    rle = cocomask.frPyObjects(seg_obj, h, w)
    mask = cocomask.decode(rle)
    return mask


def rle_to_mask(rle: dict) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def load_json(path):
    with open(path, "r") as f:
        info = yaml.load(f, Loader=yaml.CLoader)
    return info


def save_json(path, info):
    # save to json without sorting keys or changing format
    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def cast_formats_for_json(data):
    # casting for every keys in dict to list so that it can be saved as json
    for key in data.keys():
        if (
            isinstance(data[key][0], np.ndarray)
            or isinstance(data[key][0], np.float32)
            or isinstance(data[key][0], np.float64)
            or isinstance(data[key][0], np.int32)
            or isinstance(data[key][0], np.int64)
        ):
            data[key] = np.array(data[key]).tolist()
    return data


def read_preds(pred_dir):
    rgb_paths = sorted(glob(f"{pred_dir}/rgb/*"))
    poses_paths = sorted(glob(f"{pred_dir}/poses/*"))
    poses_gt_paths = sorted(glob(f"{pred_dir}/poses_gt/*"))
    rgbs = []
    poses = []
    poses_gt = []
    frame_ids = []
    for i in range(len(rgb_paths)):
        rgb_path = rgb_paths[i]
        pose_path = poses_paths[i]
        pose_gt_path = poses_gt_paths[i]
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pose = load_pose(pose_path)
        pose_gt = load_pose(pose_gt_path)
        frame_id = Path(rgb_path).stem
        rgbs.append(rgb)
        poses.append(pose)
        poses_gt.append(pose_gt)
        frame_ids.append(frame_id)
    return {"rgbs": rgbs, "poses": poses, "frame_ids": frame_ids, "poses_gt": poses_gt}


def load_pose(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".txt"):
        return np.loadtxt(path).reshape(4, 4)
    else:
        raise ValueError(f"Unknown pose format: {path}")


def load_depth_(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)


def load_mask_(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def load_rgb_(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def load_depth(path, wh=None, zfar=np.inf, do_convert_to_m=True):
    depth = load_depth_(path)
    if do_convert_to_m:
        depth = depth.astype(np.float32) / 1e3
    if wh is not None:
        depth = resize_img(depth, wh=wh)
    depth[(depth < 0.001) | (depth >= zfar)] = 0
    return depth


def resize_img(depth, wh):
    return cv2.resize(depth, (wh[0], wh[1]), interpolation=cv2.INTER_NEAREST)


def load_color(path, wh=None):
    color = load_rgb_(path)
    if wh is not None:
        color = resize_img(color, wh=wh)
    return color


def load_mask(path, wh=None):
    mask = load_mask_(path)
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break
    if wh is not None:
        mask = resize_img(mask, wh=wh)
    mask = mask.astype(bool)
    return mask
