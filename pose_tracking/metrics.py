import copy

import numpy as np
import torch
from pose_tracking.utils.common import cast_to_numpy
from pose_tracking.utils.geom import transform_pts, world_to_cam
from pose_tracking.utils.trimesh_utils import get_posed_model_pts
from scipy.spatial import cKDTree
from sklearn import metrics


def calc_metrics(
    pred_rt,
    gt_rt,
    class_name=None,
    model=None,
    pts=None,
    bbox_3d=None,
    handle_visibility=False,
    use_miou=False,
    use_symmetry=True,
    diameter=None,
    is_meters=True,
    log_fn=print,
):
    """
    Calculate required metrics for pose estimation. Metric units are mm and degrees.
    Args:
        pred_rt: predicted pose
        gt_rt: ground truth pose
        model: CAD model of the object as a trimesh object
        pts: points of the object, if model is not provided
        class_name: name of the object (applies to Linemod)
        bbox: bounding box of the object
        handle_visibility: account for visibility of a handle (e.g., for a mug)
        use_miou: whether to include miou
        use_symmetry: whether to use symmetry when calculating miou
        diameter: diameter of the object for ADD/ADDS-10 calculation

    """
    pred_rt = cast_to_numpy(pred_rt)
    gt_rt = cast_to_numpy(gt_rt)
    if pts is not None:
        pts = cast_to_numpy(pts)
    if is_meters:
        pred_rt = copy.deepcopy(pred_rt)
        pred_rt[:3, 3] *= 1000
        gt_rt = copy.deepcopy(gt_rt)
        gt_rt[:3, 3] *= 1000
    res = {}
    try:
        add = calc_add(pred_rt, gt_rt, pts=pts, model=model)
        adds = calc_adds(pred_rt, gt_rt, pts=pts, model=model)
        res.update(
            {
                "add": add,
                "adds": adds,
            }
        )
    except Exception as e:
        log_fn(f"Error calculating metrics: {e}\nInputs:\n{pred_rt=},\n{gt_rt=},\n{pts=},\n{model=}")
        res.update(
            {
                "add": torch.nan,
                "adds": torch.nan,
                "r_err": torch.nan,
                "t_err": torch.nan,
                "5deg5cm": torch.nan,
                "2deg2cm": torch.nan,
            }
        )
        if use_miou:
            res["miou"] = torch.nan
        if np.isnan(pred_rt).any():
            log_fn("pred_rt has nan values")
        return res
    if use_miou:
        assert bbox_3d is not None
        bbox_3d = cast_to_numpy(bbox_3d)
        if is_meters:
            bbox_3d *= 1000
        miou = calc_3d_iou_new(
            pred_rt,
            gt_rt,
            bbox=bbox_3d,
            handle_visibility=handle_visibility,
            class_name=class_name,
            use_symmetry=use_symmetry,
        )
        res["miou"] = miou

    rt_errors = calc_rt_errors(pred_rt, gt_rt, class_name=class_name, handle_visibility=handle_visibility)
    res.update(rt_errors)

    if diameter is not None:
        diameter = cast_to_numpy(diameter)
        if is_meters:
            diameter *= 1000
        thresh = diameter * 0.1
        res["add10"] = add < thresh
        res["adds10"] = adds < thresh

    return res


def calc_add(pred_rt, gt_rt, pts=None, model=None):
    """
    TODO: ensure aligns with bop's (wrt using obj diameter)
    """
    assert pts is None or model is None, "Either pts or model should be provided"
    if pts is None:
        pts_pred = get_posed_model_pts(pred_rt, model)
        pts_gt = get_posed_model_pts(gt_rt, model)
    else:
        pts_pred = transform_pts(pts=pts, pose=pred_rt)
        pts_gt = transform_pts(pts=pts, pose=gt_rt)
    e = calc_add_pts(pts_pred, pts_gt)
    return e


def calc_add_pts(pts1, pts2):
    assert pts1.shape[1] == pts2.shape[1] == 3
    return np.linalg.norm(pts1 - pts2, axis=1).mean()


def calc_adds(pred_rt, gt_rt, pts=None, model=None):
    assert pts is None or model is None, "Either pts or model should be provided"
    if pts is None:
        pts_pred = get_posed_model_pts(pred_rt, model)
        pts_gt = get_posed_model_pts(gt_rt, model)
    else:
        pts_pred = transform_pts(pts=pts, pose=pred_rt)
        pts_gt = transform_pts(pts=pts, pose=gt_rt)

    e = calc_adds_pts(pts_pred, pts_gt)
    return e


def calc_adds_pts(pts_pred, pts_gt):
    nn_index = cKDTree(pts_pred)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e


def calc_auc(errs, max_val=0.1, step=0.001):

    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val)
    return {
        "auc": auc,
        "recall": Y,
        "thresholds": X,
    }


def calc_rt_errors(rt1, rt2, handle_visibility=False, class_name=""):
    """Calculate rotation and translation errors between two poses.
    Can handle symmetries in Linemod objects.
    """
    rt1 = cast_to_numpy(rt1)
    rt2 = cast_to_numpy(rt2)

    T1 = rt1[:3, 3]
    T2 = rt2[:3, 3]
    rot_pred = rt2[:3, :3]
    rot_gt = rt1[:3, :3]

    theta = calc_r_error(rot_pred, rot_gt, handle_visibility=handle_visibility, class_name=class_name)
    shift = calc_t_error(T1, T2)
    result = {"r_err": theta, "t_err": shift}

    deg_cm_errors = calc_n_deg_m_cm_errors((theta, shift))
    result.update(deg_cm_errors)

    return result


def calc_t_error(T1, T2):
    return np.linalg.norm(T1 - T2)


def calc_r_error(rot_pred, rot_gt, handle_visibility=False, class_name=""):
    R2 = normalize_rotation_matrix(rot_pred)
    R1 = normalize_rotation_matrix(rot_gt)

    if class_name in ["bottle", "can", "bowl"]:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name == "mug" and handle_visibility:
        # mag can appear symmetric when handle is not visible
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.T
        R_rot = R1 @ y_180_RT @ R2.T
        theta = min(np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = np.dot(R1, R2.T)
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    return theta


def normalize_rotation_matrix(matrix):
    U, _, Vt = np.linalg.svd(matrix)
    return np.dot(U, Vt)


def calc_n_deg_m_cm_errors(rt_error):
    # translation error is expected to be in mm
    r_geodesic, t_dist_mm = rt_error
    t_dist_cm = t_dist_mm * 0.1
    res = {}
    for r_t, t_t in [
        (5, 5),
        (2, 2),
    ]:
        name = f"{r_t}deg{t_t}cm"
        value = np.logical_and(r_geodesic <= r_t, t_dist_cm <= t_t)
        res[name] = value
    return res


def calc_3d_iou_new(rt1, rt2, bbox, handle_visibility, class_name, use_symmetry=True):
    """Computes IoU overlaps between two 3d bboxes."""

    assert bbox.shape == (8, 3) and bbox.shape == (8, 3)

    if use_symmetry and (class_name in ["bottle", "bowl", "can"]) or (class_name == "mug" and handle_visibility == 0):

        def y_rotation_matrix(theta):
            return np.array(
                [
                    [np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1],
                ]
            )

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_rt_1 = rt1 @ y_rotation_matrix(2 * np.pi * i / float(n))
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_rt_1, rt2, bbox))
    else:
        max_iou = asymmetric_3d_iou(rt1, rt2, bbox)

    return max_iou


def asymmetric_3d_iou(rt1, rt2, bbox_w):
    bbox1_c = world_to_cam(bbox_w, rt1)

    bbox2_c = world_to_cam(bbox_w, rt2)

    bbox1_max = np.amax(bbox1_c, axis=0)
    bbox1_min = np.amin(bbox1_c, axis=0)
    bbox2_max = np.amax(bbox2_c, axis=0)
    bbox2_min = np.amin(bbox2_c, axis=0)

    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)

    # intersections and union
    if np.amin(overlap_max - overlap_min) < 0:
        intersections = 0
    else:
        intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox1_max - bbox1_min) + np.prod(bbox2_max - bbox2_min) - intersections
    overlaps = intersections / union
    return overlaps


def check_spheres_overlap(t1, t2, diameter):
    center_dist = np.linalg.norm(t1 - t2)
    return center_dist < diameter


def geodesic_numpy(R1, R2):
    theta = (np.trace(R2.dot(R1.T)) - 1) / 2
    theta = np.clip(theta, -1, 1)
    return np.degrees(np.arccos(theta))


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err
