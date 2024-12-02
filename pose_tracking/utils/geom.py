import cv2
import numpy as np
import torch
from pose_tracking.config import logger
from pose_tracking.utils.misc import pick_library
from scipy.spatial.transform import Rotation as R


def world_to_2d(pts, K, rt):
    # returns N x 2 pts
    pts_cam = world_to_cam(pts, rt)
    return cam_to_2d(pts_cam, K)


def cam_to_2d(pts, K):
    # project 3d pts in camera frame onto img

    if len(pts.shape) == 3:
        new_pts = torch.bmm(K, pts.transpose(-1, -2)).transpose(-1, -2)
        new_pts = (new_pts / new_pts[..., 2:])[..., :2]
        return new_pts

    if pts.shape[-1] == 3:
        pts = pts.T
    new_pts = K @ pts
    new_pts = new_pts[:2, ...] / new_pts[2, ...]
    return new_pts.T


def world_to_cam(pts, rt):
    # returns N x 3 pts in camera frame
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    if pts.shape[1] == 3:
        pts = pts.T
    new_pts = rt[:3, :3] @ pts + rt[:3, 3].reshape(3, 1)
    return new_pts.T


def world_to_2d_pt_homo(pt, K, rt):
    if pt.shape[-1] == 4:
        pt = pt.T
    assert pt.shape[0] == 4
    pts_cam = (rt @ pt)[:3, :]
    projected = cam_to_2d(pts_cam, K).squeeze()
    return projected.round().astype(int)


def convert_3d_bbox_to_2d(bbox, intrinsics, hw, pose=None):
    pose = np.eye(4) if pose is None else pose
    bbox_2d = world_to_2d(bbox, intrinsics, rt=pose)
    u, v = bbox_2d[:, 0].astype(int), bbox_2d[:, 1].astype(int)
    h, w = hw
    x_min, y_min = np.min(u), np.min(v)
    x_max, y_max = np.max(u), np.max(v)
    x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
    bbox_2d = np.array([[x_min, y_min], [x_max, y_max]])
    return bbox_2d


def get_34_intrinsics(K):
    return np.hstack([K, np.zeros((3, 1))])


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


def backproj_2d_to_3d_batch(pts, depth, K):
    return [backproj_2d_to_3d(pt, d, Ki) for pt, d, Ki in zip(pts, depth, K)]


def backproj_2d_to_3d(pts, depth, K):
    """
    Backproject 2D points to 3D points
    Args:
        pts: (N, 2) or (2, N)
        depth: 1D depth values
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    lib = pick_library(pts)
    if pts.shape[1] != 2:
        pts = pts.T
    ones = lib.ones((pts.shape[0], 1))
    if lib == torch:
        ones = ones.to(pts.device)
    pts = lib.hstack((pts, ones))
    pts = pts * depth.reshape(-1, 1)
    pts = lib.linalg.inv(K) @ pts.T
    return pts.T


def pose_to_egocentric_delta_pose(prev_pose_mat, cur_pose_mat):
    """Extract r and t deltas from two poses in the same frame"""
    trans_delta = cur_pose_mat[:, :3, 3] - prev_pose_mat[:, :3, 3]
    rot_mat_delta = cur_pose_mat[:, :3, :3] @ prev_pose_mat[:, :3, :3].permute(0, 2, 1)
    return trans_delta, rot_mat_delta


def egocentric_delta_pose_to_pose(prev_pose_mat, trans_delta, rot_mat_delta, do_couple_rot_t=False):
    """Infer a new pose from a pose and deltas"""
    cur_pose_mat = (
        torch.eye(4, dtype=torch.float, device=prev_pose_mat.device)[None]
        .expand(len(prev_pose_mat), -1, -1)
        .contiguous()
    )
    if do_couple_rot_t:
        # both deltas are in the global frame
        cur_pose_mat[:, :3, 3] = (
            torch.bmm(rot_mat_delta, prev_pose_mat[:, :3, 3].unsqueeze(-1)).squeeze(-1) + trans_delta
        )
    else:
        cur_pose_mat[:, :3, 3] = prev_pose_mat[:, :3, 3] + trans_delta
    cur_pose_mat[:, :3, :3] = rot_mat_delta @ prev_pose_mat[:, :3, :3]
    return cur_pose_mat


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
    pose_dim = len(pose.shape)
    if pose_dim == 4:
        pts = pts[:, None]
        assert pose.shape[-2:] == (4, 4)
    elif pose_dim == 3:
        assert pose.shape == (bsz, 4, 4)
    else:
        raise ValueError("Unsupported shape for T", pose.shape)
    pts = pts[..., None]
    pose = pose[None] if pose_dim == 4 else pose[:, None]
    pts_transformed = pose[..., :3, :3] @ pts + pose[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def transform_pts(pts, r=None, t=None, pose=None):
    """
    Applies a rigid transformation to 3D points.

    Args:
        pts: nx3 ndarray with 3D points.
        r: 3x3 rotation matrix.
        t: 3x1 translation vector.
    Returns:
        nx3 ndarray with transformed 3D points.
    """
    assert (r is not None and t is not None) or pose is not None, "Either r and t or pose should be provided"
    assert pts.shape[1] == 3
    if pose is not None:
        r = pose[:3, :3]
        t = pose[:3, 3]
    pts_t = r @ pts.T + t.reshape((3, 1))
    return pts_t.T


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
