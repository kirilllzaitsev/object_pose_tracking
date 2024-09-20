import json

import numpy as np
import yaml
from pycocotools import mask as cocomask


def parse_rle_mask(seg_obj):
    h, w = seg_obj["size"]
    rle = cocomask.frPyObjects(seg_obj, h, w)
    mask = cocomask.decode(rle)
    return mask


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
