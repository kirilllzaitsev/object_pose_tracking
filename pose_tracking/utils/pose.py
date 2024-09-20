import numpy as np


def combine_R_and_T(R, T, scale_translation=1.0):
    matrix4x4 = np.eye(4)
    matrix4x4[:3, :3] = np.array(R).reshape(3, 3)
    matrix4x4[:3, 3] = np.array(T).reshape(-1) * scale_translation
    return matrix4x4
