import functools
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
from pose_tracking.config import logger
from pose_tracking.utils.common import cast_to_numpy
from pose_tracking.utils.misc import is_empty, is_tensor, pick_library
from pose_tracking.utils.pose import matrix_to_quaternion
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from scipy.spatial.transform import Rotation as R

try:
    from transforms3d.axangles import axangle2mat
    from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat
except ImportError:
    print("Warning: transforms3d not installed. Some functions may not work.")


def world_to_2d(pts, K, rt):
    # returns N x 2 pts
    pts_cam = world_to_cam(pts, rt)
    return cam_to_2d(pts_cam, K)


def cam_to_2d(pts, K):
    # project 3d pts in camera frame onto img

    t_func = get_transpose_func(pts)
    if len(pts.shape) > len(K.shape):
        K = K[None]
    pts_T = t_func(pts) if pts.shape[-1] == 3 else pts
    new_pts = K @ pts_T
    new_pts = new_pts[..., :2, :] / new_pts[..., 2:, :]
    return t_func(new_pts)


def world_to_cam(pts, rt):
    # returns (B x) N x 3 pts in camera frame
    return transform_pts(pts, rt=rt)


def backproj_2d_pts_world(pts, depth, K, rt):
    pts_cam = backproj_2d_pts(pts, depth, K)
    return world_to_cam(pts_cam, rt)


def get_transpose_func(x, dim0=-1, dim1=-2):
    if is_tensor(x):
        t_func = functools.partial(torch.transpose, dim0=dim0, dim1=dim1)
    else:
        t_func = functools.partial(np.swapaxes, axis1=dim0, axis2=dim1)
    return t_func


def world_to_2d_pt_homo(pt, K, rt):
    if pt.shape[-1] == 4:
        pt = pt.T
    assert pt.shape[0] == 4
    pts_cam = (rt @ pt)[:3, :]
    projected = cam_to_2d(pts_cam, K).squeeze()
    return projected.round().astype(int)


def convert_3d_bbox_to_2d(bbox, intrinsics, hw, pose=None):
    # returns bbox in (ul, br) format
    if len(bbox.shape) == 3:
        return np.stack(
            [
                convert_3d_bbox_to_2d(b, intrinsics, hw, pose=pose if pose is None else pose[i])
                for i, b in enumerate(bbox)
            ]
        )
    pose = np.eye(4) if pose is None else pose
    bbox_2d = world_to_2d(bbox, intrinsics, rt=pose)
    bbox_2d = cast_to_numpy(bbox_2d)
    u, v = bbox_2d[:, 0].astype(int), bbox_2d[:, 1].astype(int)
    h, w = hw
    x_min, y_min = np.min(u), np.min(v)
    x_max, y_max = np.max(u), np.max(v)
    bbox_2d = np.array([[x_min, y_min], [x_max, y_max]])
    return bbox_2d


def get_34_intrinsics(K):
    return np.hstack([K, np.zeros((3, 1))])


def get_inv_pose(pose=None, rot=None, t=None):
    if pose is not None:
        assert rot is None and t is None
        rot = pose[..., :3, :3]
        t = pose[..., :3, 3]
    lib = pick_library(t)
    inv_pose = lib.eye(4)
    if is_tensor(t):
        inv_pose = inv_pose.to(t.device)
        inv_rot = rot.transpose(-1, -2)
    else:
        inv_rot = rot.swapaxes(-1, -2)
    if len(rot.shape) == 3:
        inv_pose = inv_pose[None].repeat(len(rot), 1, 1)
    inv_pose[..., :3, :3] = inv_rot
    inv_pose[..., :3, 3] = -lib.einsum("...ij,...j->...i", inv_rot, t)
    return inv_pose


def get_pose(rot, t):
    lib = pick_library(rot)
    pose = lib.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = t
    return pose


def backproj_depth(depth, intrinsics, instance_mask=None, do_flip_xy=True):
    """
    Backproject depth for selected pixels
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    val_depth = depth > 0
    if instance_mask is None:
        val_mask = val_depth
    else:
        val_mask = np.logical_and(instance_mask, val_depth)
    idxs = np.where(val_mask)
    grid = np.array([idxs[1], idxs[0]])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, n]
    xyz = intrinsics_inv @ uv_grid
    xyz = np.transpose(xyz).squeeze()

    # rescale
    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]

    if do_flip_xy:
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
    return pts, idxs


def backproj_depth_torch(depth, intrinsics, instance_mask=None, do_flip_xy=True):
    """
    Backproject depth for selected pixels
    """
    device = depth.device
    dtype = depth.dtype

    # Invert intrinsics
    intrinsics_inv = torch.inverse(intrinsics)

    # Create a valid depth mask
    val_depth = depth > 0
    if instance_mask is None:
        val_mask = val_depth
    else:
        val_mask = torch.logical_and(instance_mask, val_depth)

    # Get indices of valid pixels
    idxs = torch.nonzero(val_mask, as_tuple=True)

    # Create UV grid
    u = idxs[1].to(dtype)
    v = idxs[0].to(dtype)
    ones = torch.ones_like(u, device=device, dtype=dtype)
    uv_grid = torch.stack([u, v, ones], dim=0)  # [3, N]

    # Backproject to 3D space
    xyz = torch.matmul(intrinsics_inv, uv_grid)  # [3, N]
    xyz = xyz.T  # [N, 3]

    # Rescale by depth values
    z = depth[idxs[0], idxs[1]]  # [N]
    pts = xyz * (z[:, None] / xyz[:, -1:])  # Rescale using depth

    # Flip x and y axes if needed
    if do_flip_xy:
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

    return pts, idxs


def calibrate_2d_pts_batch(pts, K):
    return torch.stack([calibrate_2d_pts(p, K[i]) for i, p in enumerate(pts)]).to(pts.device)


def calibrate_2d_pts(pts, K):
    """
    Calibrate 2D points to 3D points
    Args:
        pts: (N, 2) or (2, N)
        K: 3x3 camera intrinsic matrix
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    lib = pick_library(pts)
    if pts.shape[1] != 2:
        pts = pts.T
    ones = lib.ones((pts.shape[0], 1))
    if lib == torch:
        ones = ones.to(pts.device)
    pts = lib.hstack((pts, ones))
    pts = lib.linalg.inv(K) @ pts.T
    pts = pts[:2, ...] / pts[2, ...]
    return pts.T


def pose_to_egocentric_delta_pose(prev_pose_mat, cur_pose_mat):
    """Extract r and t deltas from two poses in the same frame"""
    trans_delta = cur_pose_mat[..., :3, 3] - prev_pose_mat[..., :3, 3]

    if is_tensor(prev_pose_mat):
        prev_rot_mat_inv = prev_pose_mat[..., :3, :3].transpose(-2, -1)
    else:
        prev_rot_mat_inv = prev_pose_mat[..., :3, :3].swapaxes(-2, -1)

    rot_mat_delta = cur_pose_mat[..., :3, :3] @ prev_rot_mat_inv
    return trans_delta, rot_mat_delta


def pose_to_egocentric_delta_pose_mat(prev_pose_mat, cur_pose_mat):
    t, rot = pose_to_egocentric_delta_pose(prev_pose_mat, cur_pose_mat)
    lib = pick_library(t)
    pose = lib.eye(4)
    if len(t.shape) == 2:
        pose = pose[None].repeat(len(t), 1, 1)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = t
    return pose


def egocentric_delta_pose_to_pose(
    prev_pose_mat, trans_delta=None, rot_mat_delta=None, pose_delta=None, do_couple_rot_t=False
):
    """Infer a new pose from a pose and deltas"""
    assert (trans_delta is not None and rot_mat_delta is not None) or pose_delta is not None, pose_delta
    if pose_delta is not None:
        assert trans_delta is None and rot_mat_delta is None
        trans_delta = pose_delta[..., :3, 3]
        rot_mat_delta = pose_delta[..., :3, :3]
    cur_pose_mat = (
        torch.eye(4, dtype=torch.float, device=prev_pose_mat.device) if is_tensor(prev_pose_mat) else np.eye(4)
    )
    if prev_pose_mat.ndim == 3:
        cur_pose_mat = cur_pose_mat[None].repeat(len(prev_pose_mat), 1, 1)
    if prev_pose_mat.ndim == 4:
        cur_pose_mat = cur_pose_mat[None][None].repeat(len(prev_pose_mat), len(prev_pose_mat[0]), 1, 1)
    if do_couple_rot_t:
        # both deltas are in the global frame
        cur_pose_mat[..., :3, 3] = (
            torch.bmm(rot_mat_delta, prev_pose_mat[..., :3, 3].unsqueeze(-1)).squeeze(-1) + trans_delta
        )
    else:
        cur_pose_mat[..., :3, 3] = prev_pose_mat[..., :3, 3] + trans_delta
    cur_pose_mat[..., :3, :3] = rot_mat_delta @ prev_pose_mat[..., :3, :3]
    return cur_pose_mat


def convert_rt_to_r_t(rt):
    r = rt[..., :3, :3]
    t = rt[..., :3, 3]
    return r, t


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def to_homo_torch(pts):
    """
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    """
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones), dim=-1)
    return homo


def perspective(K, obj_pose, pts):
    results = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        R, T = obj_pose[:3, :3], obj_pose[:3, 3]
        rep = np.matmul(K, np.matmul(R, pts[i].reshape(3, 1)) + T.reshape(3, 1))
        results[i, 0] = np.int32(rep[0] / rep[2])  # as matplot flip  x axis
        results[i, 1] = np.int32(rep[1] / rep[2])
    return results


def crop_frame(
    img,
    mask,
    intrinsic,
    openCV_pose,
    image_size,
    keep_inplane=False,
    virtual_bbox_size=0.3,
):
    origin_obj = np.array([0, 0, 0, 1.0])
    origin_in_cam = np.dot(openCV_pose, origin_obj)[0:3]  # center pt in camera space
    if keep_inplane:
        upper = np.array([0.0, -origin_in_cam[2], origin_in_cam[1]])
        right = np.array(
            [
                origin_in_cam[1] * origin_in_cam[1] + origin_in_cam[2] * origin_in_cam[2],
                -origin_in_cam[0] * origin_in_cam[1],
                -origin_in_cam[0] * origin_in_cam[2],
            ]
        )
        if np.linalg.norm(upper) == 0 and np.linalg.norm(right) == 0:
            logger.warning("upper and right are both zero")
            upper = np.array([0, -1, 0])
            right = np.array([1, 0, 0])
    else:
        upV = np.array([0, 0, 6]) - origin_in_cam
        upV = (np.dot(openCV_pose, [upV[0], upV[1], upV[2], 1]))[0:3]
        right = np.cross(origin_in_cam, upV)
        upper = np.cross(right, origin_in_cam)
        if np.linalg.norm(upper) == 0 and np.linalg.norm(right) == 0:
            upper = np.array([0, -1, 0])
            right = np.array([1, 0, 0])

    upper = upper * (virtual_bbox_size / 2) / np.linalg.norm(upper)
    right = right * (virtual_bbox_size / 2) / np.linalg.norm(right)

    # world coord of corner points
    w1 = origin_in_cam + upper - right
    w2 = origin_in_cam - upper - right
    w3 = origin_in_cam + upper + right
    w4 = origin_in_cam - upper + right

    # coord of corner points on image plane
    virtual_bbox = np.concatenate(
        (
            w1.reshape((1, 3)),
            w2.reshape((1, 3)),
            w3.reshape((1, 3)),
            w4.reshape((1, 3)),
        ),
        axis=0,
    )
    virtual_bbox2d = perspective(intrinsic, np.eye(4), virtual_bbox)
    virtual_bbox2d = virtual_bbox2d.astype(np.int32)
    target_virtual_bbox2d = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32) * image_size
    M = cv2.getPerspectiveTransform(virtual_bbox2d.astype(np.float32), target_virtual_bbox2d)
    cropped_img = cv2.warpPerspective(np.asarray(img), M, (image_size, image_size))
    if mask is not None:
        cropped_mask = cv2.warpPerspective(np.asarray(mask), M, (image_size, image_size))
        return cropped_img, cropped_mask
    else:
        return cropped_img


def render_pts_to_image(cvImg, meshPts, K, openCV_obj_pose, color):
    """
    Renders 3D points onto an image.

    Parameters:
        cvImg: the input image on which to render the points
        meshPts: nx3 array of 3D points in object coordinate frame
        K: 3x3 camera intrinsic matrix
        openCV_obj_pose: 4x4 homogeneous transformation matrix from object to camera frame
        color: color to render the points (B, G, R)

    Returns:
        The image with the points rendered onto it.
    """

    assert meshPts.shape[1] == 3, "meshPts should be an Nx3 array."

    # to homo
    ones = np.ones((meshPts.shape[0], 1))
    meshPts_homogeneous = np.hstack((meshPts, ones))

    # world->camera frame
    meshPts_camera_homogeneous = (openCV_obj_pose @ meshPts_homogeneous.T).T

    meshPts_camera = meshPts_camera_homogeneous[:, :3]

    Xc = meshPts_camera[:, 0]
    Yc = meshPts_camera[:, 1]
    Zc = meshPts_camera[:, 2]

    # pts in front of the camera
    valid_idx = Zc > 0
    Xc = Xc[valid_idx]
    Yc = Yc[valid_idx]
    Zc = Zc[valid_idx]

    points_camera = np.vstack((Xc, Yc, Zc))

    # 2d proj
    points_image_homogeneous = K @ points_camera

    w = points_image_homogeneous[2, :]
    u = points_image_homogeneous[0, :] / w
    v = points_image_homogeneous[1, :] / w

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # fit within bounds
    height, width = cvImg.shape[:2]
    valid_u = np.logical_and(u >= 0, u < width)
    valid_v = np.logical_and(v >= 0, v < height)
    valid = np.logical_and(valid_u, valid_v)

    u = u[valid]
    v = v[valid]

    cvImg[v, u] = color

    return cvImg


def backproj_2d_pts(pts, depth, K):
    """
    Backproject 2D points to 3D points
    Args:
        pts: optional (B,) + (N, 2) or (2, N)
        K: 3x3 camera intrinsic matrix
        depth: 1D depth values
    """
    t_func = get_transpose_func(pts)
    if pts.shape[-1] != 2:
        pts = t_func(pts)
    lib = pick_library(pts)
    ones = lib.ones((*pts.shape[:-1], 1))
    if is_tensor(pts):
        ones = ones.to(pts.device)
        pts = lib.cat((pts, ones), dim=-1)
    else:
        pts = lib.concatenate((pts, ones), axis=-1)
    depth_broadcast = depth.squeeze()[..., None]
    pts = pts * depth_broadcast
    pts_T = t_func(pts)
    pts = lib.linalg.inv(K) @ pts_T
    return t_func(pts)


def look_at_rotation(point):
    """
    @param point: point in normalized image coordinates not in pixels
    @return: R
    R @ x_raw -> x_lookat
    """
    x, y = point
    R1 = R.from_euler("xyz", [-np.arctan2(x, 1), 0, 0], degrees=False).as_matrix()
    R2 = R.from_euler("xyz", [np.arctan2(y, 1), 0, 0], degrees=False).as_matrix()
    return R2 @ R1


def transform_pts_batch(pts: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pose: (bsz, 4, 4) or (bsz, dim2, 4, 4)
        pts: (bsz, n_pts, 3)
    """
    return transform_pts(pts, rt=pose)


def transform_pts(pts, r=None, t=None, rt=None):
    """
    Returns:
        nx3 ndarray with transformed 3D points.
    """
    if rt is not None:
        r = rt[..., :3, :3]
        t = rt[..., :3, 3]
    t_func = get_transpose_func(pts)
    if pts.shape[-1] == 3:
        pts = t_func(pts)
    new_pts = r @ pts + t[..., None]
    return t_func(new_pts)


def rotate_pts(pts, r):
    t_func = get_transpose_func(pts)
    if pts.shape[-1] == 3:
        pts = t_func(pts)
    new_pts = r @ pts
    return t_func(new_pts)


def rotate_pts_batch(R, pts):
    return torch.bmm(R, pts.transpose(-1, -2)).transpose(-1, -2)


def zoom_in(im, c, s, res, channel=3, interpolate=cv2.INTER_LINEAR):
    """
    zoom in on the object with center c and size s, and resize to resolution res.
    :param im: nd.array, single-channel or 3-channel image
    :param c: (w, h), object center
    :param s: scalar, object size
    :param res: target resolution
    :param channel:
    :param interpolate:
    :return: zoomed object patch
    """
    c_w, c_h = c
    c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
    if channel == 1:
        im = im[..., None]
    h, w = im.shape[:2]
    u = int(c_h - 0.5 * s + 0.5)
    l = int(c_w - 0.5 * s + 0.5)
    b = u + s
    r = l + s
    if (u >= h) or (l >= w) or (b <= 0) or (r <= 0):
        return np.zeros((res, res, channel)).squeeze()
    if u < 0:
        local_u = -u
        u = 0
    else:
        local_u = 0
    if l < 0:
        local_l = -l
        l = 0
    else:
        local_l = 0
    if b > h:
        local_b = s - (b - h)
        b = h
    else:
        local_b = s
    if r > w:
        local_r = s - (r - w)
    else:
        local_r = s
    im_crop = np.zeros((s, s, channel))
    im_crop[local_u:local_b, local_l:local_r, :] = im[u:b, l:r, :]
    im_crop = im_crop.squeeze()
    im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
    c_h = 0.5 * (u + b)
    c_w = 0.5 * (l + r)
    s = s
    return im_resize, c_h, c_w, s


def interpolate_bbox_edges(corners, num_points=24):
    """
    Interpolate points uniformly between corners of a 3D bounding box.

    Args:
        corners: Array of shape (8, 3), representing the 8 corners of the bbox.
        num_points: Total number of interpolated points to generate.

    Returns:
        Array of shape (num_points, 3), containing the interpolated points.
    """

    if isinstance(corners, list) or len(corners.shape) == 3:
        # return np.stack([interpolate_bbox_edges(c, num_points) for c in corners])
        res = defaultdict(list)
        for i, c in enumerate(corners):
            for k, v in interpolate_bbox_edges(c, num_points).items():
                res[k].append(v)
        return {k: np.stack(v) for k, v in res.items()}

    # Define the 12 edges of the 3D bounding box by corner indices
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),  # Left face
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),  # Right face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # Vertical edges
    ]
    kpt_idxs = list(range(8))
    kpt_count = len(kpt_idxs)
    kpt_idx = kpt_count
    collinear_quad_idxs = []

    # Calculate number of points per edge (24 / 12 = 2 points per edge)
    points_per_edge = num_points // len(edges)
    interpolated_points = []

    for start, end in edges:
        # Get start and end corners of the edge
        p1 = corners[start]
        p2 = corners[end]
        p1_idx = start
        p2_idx = end

        # Interpolate points between p1 and p2
        t_vals = np.linspace(0, 1, points_per_edge + 2)[1:-1]  # Exclude exact corners
        edge_points = [(1 - t) * p1 + t * p2 for t in t_vals]
        edge_idxs = [p1_idx]
        for edge_pt in edge_points:
            kpt_idxs.append(kpt_idx)
            edge_idxs.append(kpt_idx)
            kpt_idx = len(kpt_idxs)
        edge_idxs.append(p2_idx)

        interpolated_points.extend(edge_points)
        collinear_quad_idxs.append(edge_idxs)

    all_points = np.concatenate([corners, interpolated_points])

    return {
        "interpolated_points": np.array(interpolated_points),
        "collinear_quad_idxs": collinear_quad_idxs,
        "kpt_idxs": kpt_idxs,
        "all_points": all_points,
    }


def rot_mat_from_6d(poses):
    """
    Computes rotation matrix from 6D continuous space according to the parametrisation proposed in
    On the Continuity of Rotation Representations in Neural Networks
    https://arxiv.org/pdf/1812.07035.pdf
    :param poses: [B, 6]
    :return: R: [B, 3, 3]
    """

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = x_raw / torch.norm(x_raw, dim=1, keepdim=True)
    z = torch.cross(x, y_raw, dim=1)
    z = z / torch.norm(z, dim=1, keepdim=True)
    y = torch.cross(z, x, dim=1)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)
    return matrix


def bbox_to_8_point_centered(min_coords=None, max_coords=None, center=None, bbox=None):
    if bbox is not None:
        min_coords = bbox["min"]
        max_coords = bbox["max"]
        center = bbox["center"]
    points = [
        [min_coords[0] - center[0], min_coords[1] - center[1], min_coords[2] - center[2]],
        [min_coords[0] - center[0], min_coords[1] - center[1], max_coords[2] - center[2]],
        [min_coords[0] - center[0], max_coords[1] - center[1], min_coords[2] - center[2]],
        [min_coords[0] - center[0], max_coords[1] - center[1], max_coords[2] - center[2]],
        [max_coords[0] - center[0], min_coords[1] - center[1], min_coords[2] - center[2]],
        [max_coords[0] - center[0], min_coords[1] - center[1], max_coords[2] - center[2]],
        [max_coords[0] - center[0], max_coords[1] - center[1], min_coords[2] - center[2]],
        [max_coords[0] - center[0], max_coords[1] - center[1], max_coords[2] - center[2]],
    ]

    return np.array(points)


def convert_3d_t_for_2d(t_gt_abs, intrinsics, hw):
    t_gt_2d = cam_to_2d(t_gt_abs.unsqueeze(-1), intrinsics).squeeze(-1)
    t_gt_2d_norm = normalize_2d_kpts(t_gt_2d, hw)

    depth_gt = t_gt_abs[..., 2]
    return t_gt_2d_norm, depth_gt


def normalize_2d_kpts(t_gt_2d, hw):
    t_gt_2d_norm = t_gt_2d.clone()
    t_gt_2d_norm[..., 0] = t_gt_2d_norm[..., 0] / hw[1]
    t_gt_2d_norm[..., 1] = t_gt_2d_norm[..., 1] / hw[0]
    return t_gt_2d_norm


def convert_2d_t_to_3d(t_pred, depth_pred, intrinsics, hw=None):
    res = {}
    # abs 2d center to abs 3d center
    t_pred_2d_denorm = t_pred.detach().clone()
    if hw is not None:
        h, w = hw.unbind(-1) if is_tensor(hw) else hw
        t_pred_2d_denorm[:, 0] = t_pred_2d_denorm[:, 0] * w
        t_pred_2d_denorm[:, 1] = t_pred_2d_denorm[:, 1] * h

    t_pred_2d_backproj = []
    for sample_idx in range(len(t_pred)):
        t_pred_2d_backproj.append(
            backproj_2d_pts(
                t_pred_2d_denorm[sample_idx][None],
                depth=depth_pred[sample_idx],
                K=intrinsics[sample_idx],
            ).squeeze()
        )
    t_pred = torch.stack(t_pred_2d_backproj).to(depth_pred.device)
    res["t_pred_2d_denorm"] = t_pred_2d_denorm
    res["t_pred"] = t_pred
    return res


def compute_normals(depth):
    # https://answers.opencv.org/question/82453/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-product/
    depth = cast_to_numpy(depth).squeeze()
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    f1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32) / 8.0
    f2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0

    f1m = cv2.flip(f1, 0)
    f2m = cv2.flip(f2, 1)

    n1 = cv2.filter2D(depth, -1, f1m, borderType=cv2.BORDER_CONSTANT)
    n2 = cv2.filter2D(depth, -1, f2m, borderType=cv2.BORDER_CONSTANT)

    n1 = -n1
    n2 = -n2

    temp = np.sqrt(n1 * n1 + n2 * n2 + 1)

    N3 = 1 / temp
    N1 = n1 * N3
    N2 = n2 * N3

    surface_normals = cv2.merge([N1, N2, N3])
    return surface_normals


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    """https://github.com/THU-DA-6D-Pose-Group/GDR-Net/blob/main/core/gdrn_modeling/models/GDRN.py"""
    # Compute rotation between ray to object centroid and optical center ray
    if is_empty(allo_pose):
        return allo_pose
    if allo_pose.ndim == 3:
        return np.stack(
            [allocentric_to_egocentric(a, src_type=src_type, dst_type=dst_type, cam_ray=cam_ray) for a in allo_pose]
        )
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((4, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((4, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    if src_type == "mat":
        allo_pose[3, 3] = 1
    return allo_pose


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.

    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = a[:, :, 1] * b[:, :, 0] + a[:, :, 2] * b[:, :, 3] - a[:, :, 3] * b[:, :, 2] + a[:, :, 0] * b[:, :, 1]
    y = -a[:, :, 1] * b[:, :, 3] + a[:, :, 2] * b[:, :, 0] + a[:, :, 3] * b[:, :, 1] + a[:, :, 0] * b[:, :, 2]
    z = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] + a[:, :, 0] * b[:, :, 3]
    w = -a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 3] + a[:, :, 0] * b[:, :, 0]

    return torch.stack((w, x, y, z), dim=2)


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[:, 0]  # Remove added Corner dimension here.
    return q_ego


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: Nx3
    # rot_allo: Nx3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quaternion_to_matrix(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def bbox_from_corners(corners):  # corners [[3], [3]] or [Bs, 2, 3]
    if not isinstance(corners, np.ndarray):
        corners = np.array(corners)

    # bbox = np.zeros((8, 3))
    bbox_shape = corners.shape[:-2] + (8, 3)  # [Bs, 8, 3]
    bbox = np.zeros(bbox_shape)
    for i in range(8):
        x, y, z = (i % 4) // 2, i // 4, i % 2
        bbox[..., i, 0] = corners[..., x, 0]
        bbox[..., i, 1] = corners[..., y, 1]
        bbox[..., i, 2] = corners[..., z, 2]
    return bbox
