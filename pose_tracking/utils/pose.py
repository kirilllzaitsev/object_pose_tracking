import numpy as np
import torch
import trimesh
from bop_toolkit_lib.transform import euler_matrix
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix


def combine_R_and_T(R, T, scale_translation=1.0):
    matrix4x4 = np.eye(4)
    matrix4x4[:3, :3] = np.array(R).reshape(3, 3)
    matrix4x4[:3, 3] = np.array(T).reshape(-1) * scale_translation
    return matrix4x4


def convert_pose_quaternion_to_matrix(pose):
    t = pose[:3].detach().cpu()
    q = pose[3:].detach().cpu()
    pose_matrix = torch.eye(4)
    pose_matrix[:3, :3] = quaternion_to_matrix(q)
    pose_matrix[:3, 3] = t
    return pose_matrix


def sample_views_icosphere(n_views, subdivisions=None, radius=1):
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while 1:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]  # (N,3)
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def symmetry_tfs_from_info(info, rot_angle_discrete=5):
    symmetry_tfs = [np.eye(4)]
    if "symmetries_discrete" in info:
        tfs = np.array(info["symmetries_discrete"]).reshape(-1, 4, 4)
        tfs[..., :3, 3] *= 0.001
        symmetry_tfs = [np.eye(4)]
        symmetry_tfs += list(tfs)
    if "symmetries_continuous" in info:
        axis = np.array(info["symmetries_continuous"][0]["axis"]).reshape(3)
        offset = info["symmetries_continuous"][0]["offset"]
        rxs = [0]
        rys = [0]
        rzs = [0]
        if axis[0] > 0:
            rxs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[1] > 0:
            rys = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[2] > 0:
            rzs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        for rx in rxs:
            for ry in rys:
                for rz in rzs:
                    tf = euler_matrix(rx, ry, rz)
                    tf[:3, 3] = offset
                    symmetry_tfs.append(tf)
    if len(symmetry_tfs) == 0:
        symmetry_tfs = [np.eye(4)]
    symmetry_tfs = np.array(symmetry_tfs)
    return symmetry_tfs
