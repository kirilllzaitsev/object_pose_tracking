import cv2
import numpy as np
import torch
from pose_tracking.config import logger


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def backproj_depth(depth, intrinsics, instance_mask, do_flip_xy=True):
    """
    Backproject depth for selected pixels
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    val_depth = depth > 0
    val_instance_mask = np.logical_and(instance_mask, val_depth)
    idxs = np.where(val_instance_mask)
    grid = np.array([idxs[1], idxs[0]])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, n]
    xyz = intrinsics_inv @ uv_grid
    xyz = np.transpose(xyz)
    
    # rescale
    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]

    if do_flip_xy:
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
    return pts, idxs


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
