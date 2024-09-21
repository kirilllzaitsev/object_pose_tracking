import copy

import numpy as np
from pose_tracking.utils.geom import world_to_cam
from scipy.spatial import cKDTree
from sklearn import metrics


def calc_metrics(
    pred,
    gt,
    model,
    class_name,
    bbox=None,
    handle_visibility=False,
    use_miou=False,
    use_symmetry=True,
):
    """
    Calculate required metrics for pose estimation
    """
    add = calc_add(pred, gt, model)
    adds = calc_adds(pred, gt, model)
    res = {
        "add": add,
        "adds": adds,
    }
    if use_miou:
        assert bbox is not None and class_name is not None
        miou = calc_3d_iou_new(
            pred,
            gt,
            bbox1=bbox,
            bbox2=bbox,
            handle_visibility=handle_visibility,
            class_name=class_name,
            use_symmetry=use_symmetry,
        )
        res["miou"] = miou

    rt_errors = calc_rt_errors(pred, gt, class_name=class_name, handle_visibility=handle_visibility)
    add_rt_errors = True
    if use_miou and res["miou"] < 0.25:
        add_rt_errors = False
    if add_rt_errors:
        res.update(rt_errors)

    return res


def calc_add(pred, gt, model):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_model = copy.deepcopy(model)
    gt_model = copy.deepcopy(model)
    pred_model.transform(pred)
    gt_model.transform(gt)
    e = np.linalg.norm(np.asarray(pred_model.points) - np.asarray(gt_model.points), axis=1).mean()
    return e


def calc_adds(pred, gt, model):
    """
    @pred: 4x4 mat
    @gt:
    @model: open3d pcd model
    """
    pred_model = copy.deepcopy(model)
    gt_model = copy.deepcopy(model)
    pred_model.transform(pred)
    gt_model.transform(gt)

    nn_index = cKDTree(np.asarray(pred_model.points).copy())
    nn_dists, _ = nn_index.query(np.asarray(gt_model.points).copy(), k=1)
    e = nn_dists.mean()
    return e


def calc_auc_sklearn(errs, max_val=0.1, step=0.001):

    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val * 1)
    return auc


def calc_rt_errors(rt1, rt2, handle_visibility, class_name):

    R1 = rt1[:3, :3] / np.cbrt(np.linalg.det(rt1[:3, :3]))
    T1 = rt1[:3, 3]

    R2 = rt2[:3, :3] / np.cbrt(np.linalg.det(rt2[:3, :3]))
    T2 = rt2[:3, 3]

    if class_name in ["bottle", "can", "bowl"]:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name == "mug" and handle_visibility:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif class_name in ["phone", "eggbox", "glue"]:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2), np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2)
    result = {"r_err": theta, "t_err": shift}

    return result


def calc_3d_iou_new(rt1, rt2, bbox1, bbox2, handle_visibility, class_name, use_symmetry=True):
    """Computes IoU overlaps between two 3d bboxes."""

    assert bbox1.shape == (8, 3) and bbox2.shape == (8, 3)

    if (
        use_symmetry
        and (class_name in ["bottle", "bowl", "can"] and class_name)
        or (class_name == "mug" and class_name and handle_visibility == 0)
    ):

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
            max_iou = max(max_iou, asymmetric_3d_iou(rotated_rt_1, rt2, bbox1, bbox2))
    else:
        max_iou = asymmetric_3d_iou(rt1, rt2, bbox1, bbox2)

    return max_iou


def asymmetric_3d_iou(rt1, rt2, bbox1_w, bbox2_w):
    bbox1_c = world_to_cam(bbox1_w, rt1)

    bbox2_c = world_to_cam(bbox2_w, rt2)

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
