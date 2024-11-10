import cv2
import numpy as np
import torch
from pose_tracking.config import logger
from pose_tracking.utils.common import infer_lib
from scipy.spatial.transform import Rotation as R


def project_3d_to_2d(pt, K, rt):
    pt = pt.reshape(4, 1)
    projected = K @ ((rt @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def world_to_cam(pts, rt):
    # returns N x 3 pts in camera frame
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    if pts.shape[1] == 3:
        pts = pts.T
    pts = np.vstack([pts, np.ones((1, pts.shape[1]), dtype=np.float32)])
    new_pts = rt @ pts
    new_pts = new_pts[:3, :] / new_pts[3, :]
    return new_pts.T


def get_34_intrinsics(K):
    return np.hstack([K, np.zeros((3, 1))])


def world_to_2d(pts, K, rt):
    # returns N x 2 pts in image frame
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    if pts.shape[1] == 3:
        pts = pts.T
    new_pts = K @ (rt[:3, :3] @ pts + rt[:3, 3].reshape(3, 1))
    new_pts = new_pts[:2, :] / new_pts[2, :]
    return new_pts.T


def get_inv_pose(pose=None, rot=None, t=None):
    if pose is not None:
        assert rot is None and t is None
        rot = pose[:3, :3]
        t = pose[:3, 3]
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = rot.T
    inv_pose[:3, 3] = -rot.T @ t
    return inv_pose


def get_pose(rot, t):
    lib = infer_lib(rot)
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


def pose_to_egocentric_delta_pose(A_in_cam, B_in_cam):
    """Extract r and t deltas from two poses in the same frame"""
    trans_delta = B_in_cam[:, :3, 3] - A_in_cam[:, :3, 3]
    rot_mat_delta = B_in_cam[:, :3, :3] @ A_in_cam[:, :3, :3].permute(0, 2, 1)
    return trans_delta, rot_mat_delta


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    """Infer a new pose from a pose and deltas"""
    B_in_cam = torch.eye(4, dtype=torch.float, device=A_in_cam.device)[None].expand(len(A_in_cam), -1, -1).contiguous()
    B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
    B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
    return B_in_cam


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


def backproj_2d_pts(pts, K, depth):
    """
    Backproject 2D points to 3D points
    Args:
        pts: (N, 2) or (2, N)
        K: 3x3 camera intrinsic matrix
        depth: 1D depth values
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    if pts.shape[1] != 2:
        pts = pts.T
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts = pts * depth.reshape(-1, 1)
    pts = np.linalg.inv(K) @ pts.T
    return pts.T


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


def transform_pts_batch(pose: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pose: (bsz, 4, 4) or (bsz, dim2, 4, 4)
        pts: (bsz, n_pts, 3)
    """
    bsz = pose.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if pose.dim() == 4:
        pts = pts.unsqueeze(1)
        assert pose.shape[-2:] == (4, 4)
    elif pose.dim() == 3:
        assert pose.shape == (bsz, 4, 4)
    else:
        raise ValueError("Unsupported shape for T", pose.shape)
    pts = pts.unsqueeze(-1)
    pose = pose.unsqueeze(-3)
    pts_transformed = pose[..., :3, :3] @ pts + pose[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def transform_pts(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    Args:
        pts: nx3 ndarray with 3D points.
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
    Returns:
        nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T
