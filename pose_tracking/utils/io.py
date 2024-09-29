import json
from glob import glob

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
    rgbs = []
    poses = []
    for rgb_path, pose_path in zip(rgb_paths, poses_paths):
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pose = load_pose(pose_path)
        rgbs.append(rgb)
        poses.append(pose)
    return {"rgbs": rgbs, "poses": poses}


def load_pose(path):
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".txt"):
        return np.loadtxt(path).reshape(4, 4)
    else:
        raise ValueError(f"Unknown pose format: {path}")
