import numpy as np


def get_visib_px_num(mask):
    # todo: handle segm mask
    return mask.sum()


def calc_scale_factor_strength(x_num, min_val, max_val, p=2):
    res = 1 - ((x_num - min_val) / (max_val - min_val)) ** p
    res = 1 - 1 / (1 + np.exp(-0.01 * (x_num - (min_val + max_val) / 2)))
    res = 1 - (x_num - min_val) / (max_val - min_val)
    return res


def calc_occlusion_factor_strength(x_num, min_val, max_val, p=2):
    res = 1 - ((x_num - min_val) / (max_val - min_val)) ** p
    res = 1 - 1 / (1 + np.exp(-0.01 * (x_num - (min_val + max_val) / 2)))
    res = 1 - (x_num - min_val) / (max_val - min_val)
    return res
