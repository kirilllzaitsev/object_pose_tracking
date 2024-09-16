import cv2
import numpy as np
import torch


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
        numpy.ndarray: The image with the points rendered onto it.
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
