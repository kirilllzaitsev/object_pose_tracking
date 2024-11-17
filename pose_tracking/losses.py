import torch
from pose_tracking.utils.geom import transform_pts_batch
from pose_tracking.utils.misc import pick_library

try:
    from pose_tracking.chamfer_distance import ChamferDistance
except Exception as e:
    print(f"Failed to import a custom ChamferDistance: {e}. Loading an alternative.")
    ChamferDistance = None


def normalize_quaternion(quat):
    norm = torch.norm(quat, dim=-1, keepdim=True)
    return quat / norm


def geodesic_loss(pred_quat, true_quat):
    pred_quat = normalize_quaternion(pred_quat)
    true_quat = normalize_quaternion(true_quat)

    dot_product = torch.sum(pred_quat * true_quat, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angles = 2 * torch.acos(torch.abs(dot_product))
    return torch.mean(angles)


def quat_loss(pred_quat, true_quat, eps=1e-7):
    gt = true_quat

    est = normalize_quaternion(pred_quat)
    inn_prod = torch.mm(est, gt.t())
    inn_prod = inn_prod.diag()

    quat_loss = 1 - (inn_prod).abs().mean()
    quat_reg_loss = (1 - pred_quat.norm(dim=1).mean()).abs()

    return quat_loss, quat_reg_loss


def compute_adds_loss(pose_pred, pose_gt, points):
    assert pose_gt.dim() == 3 and pose_gt.shape[-2:] == (4, 4)
    assert pose_pred.shape[-2:] == (4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    TXO_gt_points = transform_pts_batch(pose_gt, points)
    TXO_pred_points = transform_pts_batch(pose_pred, points)
    dists_squared = (TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)) ** 2
    dists = dists_squared
    dists_norm_squared = dists_squared.sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    losses = dists_squared[ids_row, assign, ids_col].mean(dim=(-1, -2))
    return losses


def compute_add_loss(pose_pred, pose_gt, points):
    bsz = len(pose_gt)
    assert pose_pred.shape == (bsz, 4, 4) and pose_gt.shape == (bsz, 4, 4)
    assert len(points.shape) == 3 and points.shape[-1] == 3
    lib = pick_library(points)
    dists = lib.mean(lib.abs(transform_pts_batch(pose_gt, points) - transform_pts_batch(pose_pred, points)))
    return dists


def compute_chamfer_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if ChamferDistance is None:
        return torch.tensor(torch.nan)
    chamfer_dist = ChamferDistance()
    has_batch_dim = len(x.shape) == 3
    if not has_batch_dim:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    dist1, dist2 = chamfer_dist(x, y)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss
