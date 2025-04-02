import numpy as np
import torch
import trimesh
from bop_toolkit_lib.transform import euler_matrix
from pose_tracking.utils.common import istensor
from pose_tracking.utils.rotation_conversions import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


def convert_r_t_to_rt(r, t, scale_translation=1.0, rot_repr="quaternion"):
    if r.shape[-2:] != (3, 3):
        r = convert_rot_vector_to_matrix(r, rot_repr=rot_repr)
    pose = torch.eye(4, device=r.device) if istensor(r) else np.eye(4)
    if r.ndim == 3:
        if istensor(r):
            pose = pose[None].repeat(r.shape[0], 1, 1)
        else:
            pose = pose[None].repeat(r.shape[0], 0)
    pose[..., :3, :3] = r
    pose[..., :3, 3] = t * scale_translation
    return pose


def convert_pose_vector_to_matrix(pose, rot_repr="quaternion"):
    # assumes translation occupies the first 3 elements of the pose vector
    if pose.shape[-2:] == (4, 4):
        return pose
    t = pose[..., :3]
    pose_matrix = torch.eye(4, device=pose.device)
    if len(t.shape) == 2:
        pose_matrix = pose_matrix[None].repeat(t.shape[0], 1, 1)
    pose_matrix[..., :3, 3] = t

    rot = pose[..., 3:]
    rot_mat = convert_rot_vector_to_matrix(rot, rot_repr)
    pose_matrix[..., :3, :3] = rot_mat
    return pose_matrix


def convert_rot_vector_to_matrix(rot, rot_repr="quaternion"):
    if rot.shape[-1] == 4:
        rot_mat = quaternion_to_matrix(rot)
    elif rot.shape[-1] == 3:
        if rot_repr == "axis_angle":
            rot_mat = axis_angle_to_matrix(rot)
        elif rot_repr == "euler":
            rot_mat = euler_angles_to_matrix(rot, convention="XYZ")
        else:
            raise ValueError(f"Unknown rotation representation: {rot_repr}")
    elif rot.shape[-1] == 6:
        rot_mat = rotation_6d_to_matrix(rot)
    else:
        raise ValueError(f"Unknown rotation representation: {rot.shape}")
    return rot_mat


def convert_pose_matrix_to_vector(pose, rot_repr="quaternion"):
    if pose.shape[-2:] != (4, 4):
        return pose
    if not istensor(pose):
        pose = torch.tensor(pose)
    t = pose[..., :3, 3]
    rot = pose[..., :3, :3]
    if rot_repr == "quaternion":
        rot = matrix_to_quaternion(rot)
    elif rot_repr == "axis_angle":
        rot = matrix_to_axis_angle(rot)
    elif rot_repr == "rotation6d":
        rot = matrix_to_rotation_6d(rot)
    elif rot_repr == "euler":
        rot = matrix_to_euler_angles(rot, convention="XYZ")
    else:
        raise ValueError(f"Unknown rotation representation: {rot.shape}")
    return torch.cat([t, rot], dim=-1)


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
