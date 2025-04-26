import copy
from collections import defaultdict

import numpy as np
import torch
from pose_tracking.utils.common import cast_to_numpy
from pose_tracking.utils.detr_utils import postprocess_bbox
from pose_tracking.utils.geom import transform_pts, world_to_cam
from pose_tracking.utils.kpt_utils import is_torch
from pose_tracking.utils.misc import is_tensor, pick_library
from pose_tracking.utils.trimesh_utils import get_posed_model_pts
from scipy.spatial import cKDTree
from sklearn import metrics
from torchmetrics import Accuracy
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def calc_metrics(
    pred_rt,
    gt_rt,
    model=None,
    pts=None,
    bbox_3d=None,
    use_miou=False,
    use_symmetry=True,
    diameter=None,
    is_meters=True,
    log_fn=print,
    include_adds=False,
    sym_type=None,
    class_name=None,
    handle_visibility=None,
):
    """
    Calculate required metrics for pose estimation. Metric units are mm and degrees. Inputs are in m
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
        is_meters: either m or mm
    Returns:
        res: dictionary of metrics, dist metrics are in mm, angles in degrees
    """
    pred_rt = cast_to_numpy(pred_rt)
    gt_rt = cast_to_numpy(gt_rt)
    if pts is not None:
        pts = cast_to_numpy(pts)
    res = {}
    dist_scaler_to_mm = 1e3 if is_meters else 1  # m to mm
    try:
        add = calc_add(pred_rt, gt_rt, pts=pts, model=model)
        res["add"] = add * dist_scaler_to_mm
        if include_adds:
            adds = calc_adds(pred_rt, gt_rt, pts=pts, model=model)
            res["adds"] = adds * dist_scaler_to_mm
    except Exception as e:
        log_fn(f"Error calculating metrics: {e}\n{locals()=}")
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
        bbox_3d = copy.deepcopy(cast_to_numpy(bbox_3d)).squeeze()
        miou = calc_3d_iou_new(
            pred_rt,
            gt_rt,
            bbox=bbox_3d,
            sym_type=sym_type,
            use_symmetry=use_symmetry,
        )
        res["miou"] = miou

    rt_errors = calc_rt_errors(pred_rt, gt_rt, sym_type=sym_type)
    rt_errors["t_err"] *= dist_scaler_to_mm
    res.update(rt_errors)

    if diameter is not None:
        diameter = cast_to_numpy(diameter)
        thresh = diameter * 0.1
        res["add10"] = add < thresh
        if include_adds:
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
        pts_pred = transform_pts(pts=pts, rt=pred_rt)
        pts_gt = transform_pts(pts=pts, rt=gt_rt)
    e = calc_add_pts(pts_pred, pts_gt)
    return e


def calc_add_pts(pts1, pts2):
    assert pts1.shape[-1] == pts2.shape[-1] == 3
    return np.linalg.norm(pts1 - pts2, axis=-1).mean()


def calc_adds(pred_rt, gt_rt, pts=None, model=None):
    assert pts is None or model is None, "Either pts or model should be provided"
    if pts is None:
        pts_pred = get_posed_model_pts(pred_rt, model)
        pts_gt = get_posed_model_pts(gt_rt, model)
    else:
        pts_pred = transform_pts(pts=pts, rt=pred_rt)
        pts_gt = transform_pts(pts=pts, rt=gt_rt)

    e = calc_adds_pts(pts_pred, pts_gt)
    return e


def calc_adds_pts(pts_pred, pts_gt):
    nn_index = cKDTree(pts_pred)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e


def calc_auc(errs, max_val=0.1, step=0.001, do_convert_to_percent=True):
    errs = np.sort(np.array(errs))
    X = np.arange(0, max_val + step, step=step)
    Y = np.ones(len(X))
    for i, x in enumerate(X):
        y = (errs <= x).sum() / len(errs)
        Y[i] = y
        if y >= 1:
            break
    auc = metrics.auc(X, Y) / (max_val)
    if do_convert_to_percent:
        auc *= 100
    return {
        "auc": auc,
        "recall": Y,
        "thresholds": X,
    }


def calc_rt_errors(pred_rt, gt_rt, sym_type=None):
    """Calculate rotation and translation errors between two poses.
    Can handle symmetries in Linemod objects.
    """
    pred_rt = cast_to_numpy(pred_rt)
    gt_rt = cast_to_numpy(gt_rt)

    if len(pred_rt.shape) == 3:
        errors = defaultdict(list)
        for i in range(pred_rt.shape[0]):
            error = calc_rt_errors(pred_rt[i], gt_rt[i], sym_type=sym_type)
            for k, v in error.items():
                errors[k].append(v)
        return {k: np.mean(v) for k, v in errors.items()}

    T1 = pred_rt[:3, 3]
    T2 = gt_rt[:3, 3]
    rot_pred = pred_rt[:3, :3]
    rot_gt = gt_rt[:3, :3]

    theta = calc_r_error(rot_pred, rot_gt, sym_type=sym_type)
    shift = calc_t_error(T1, T2)
    result = {"r_err": theta, "t_err": shift}

    deg_cm_errors = calc_n_deg_m_cm_errors((theta, shift))
    result.update(deg_cm_errors)

    return result


def calc_t_error(T1, T2, do_reduce=True):
    if is_torch(T1):
        res = torch.linalg.norm(T1 - T2, dim=-1)
    else:
        res = np.linalg.norm(T1 - T2, axis=-1)
    if do_reduce:
        res = res.mean()
    return res


def calc_r_error(rot_pred, rot_gt, sym_type=None):
    from pose_tracking.losses import geodesic_loss_mat

    return geodesic_loss_mat(rot_pred=rot_pred, rot_gt=rot_gt, do_return_deg=True, use_eps=False, sym_type=sym_type)


def normalize_rotation_matrix(matrix):
    if is_tensor(matrix):
        U, _, Vt = torch.linalg.svd(matrix)
        return torch.matmul(U, Vt)
    else:
        U, _, Vt = np.linalg.svd(matrix)
        return np.dot(U, Vt)


def calc_n_deg_m_cm_errors(rt_error):
    # translation error is expected to be in mm
    r_geodesic, t_dist_mm = rt_error
    t_dist_cm = t_dist_mm * 0.1
    res = {}
    for r_t, t_t in [
        (15, 15),
        (10, 10),
        (5, 5),
        (2, 2),
    ]:
        name = f"{r_t}deg{t_t}cm"
        value = np.logical_and(r_geodesic <= r_t, t_dist_cm <= t_t)
        res[name] = value
    return res


def calc_3d_iou_new(rt1, rt2, bbox, sym_type=None, use_symmetry=True):
    """Computes IoU overlaps between two 3d bboxes."""

    if bbox.ndim == 3:
        bbox = bbox.squeeze(0)
    assert bbox.shape == (8, 3), bbox.shape

    if use_symmetry and sym_type == "full":

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


@torch.no_grad()
def accuracy(pred_logits, gt_labels, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if gt_labels.numel() == 0:
        return [torch.zeros([], device=pred_logits.device)]
    maxk = max(topk)
    batch_size = gt_labels.size(0)

    _, pred = pred_logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gt_labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def eval_batch_det(outs, targets, num_classes=None):
    device = outs[0]["scores"].device
    m_map = MeanAveragePrecision(extended_summary=False).to(device)
    m_iou = IntersectionOverUnion().to(device)
    use_acc = num_classes is not None
    no_obj_class_id = num_classes
    if use_acc:
        m_acc = Accuracy(task="multiclass", num_classes=num_classes + 1).to(device)
    for idx in range(len(outs)):
        res = outs[idx]
        keep = res["scores"] > res["scores_no_object"]
        pred = {k: v[keep] for k, v in res.items()}
        target = copy.deepcopy(targets[idx])
        target["boxes"] = postprocess_bbox(target["boxes"], hw=target["size"])
        m_map.update([pred], [target])
        m_iou.update([pred], [target])
        if use_acc:
            pred_labels = pred["labels"]
            gt_labels = target["labels"]
            if pred["labels"].numel() == 0:
                pred_labels = torch.zeros_like(target["labels"]) + no_obj_class_id
            elif pred["labels"].numel() > gt_labels.numel():
                gt_labels = torch.cat(
                    [
                        gt_labels,
                        torch.zeros(pred_labels.numel() - gt_labels.numel()).long().to(device) + no_obj_class_id,
                    ]
                )
            m_acc.update(pred_labels, gt_labels)
    metrics = m_map.compute()
    metrics.update(m_iou.compute())
    if use_acc:
        metrics.update({"acc": m_acc.compute()})
    return metrics


def get_rt_errors(pred_poses, gt_poses):
    # returns deg and cm
    r_errors = []
    t_errors = []
    for pred, gt in zip(pred_poses, gt_poses):

        rot1 = gt[:3, :3]
        rot2 = pred[:3, :3]
        t1 = gt[:3, 3]
        t2 = pred[:3, 3]
        r_err = calc_r_error(rot2, rot1)
        t_err = calc_t_error(t1, t2)
        r_errors.append(r_err)
        t_errors.append(t_err * 100)
    return r_errors, t_errors


def calc_metrics_agg(poses_pred, poses_gt, **kwargs):
    metrics_all = defaultdict(list)
    for pred_rt, gt_rt in zip(poses_pred, poses_gt):
        metrics = calc_metrics(pred_rt=pred_rt, gt_rt=[gt_rt], **kwargs)
        for k, v in metrics.items():
            metrics_all[k].append(v)
    metrics_all_agg = {k: np.mean(v) for k, v in metrics_all.items()}
    return metrics_all_agg
