import torch
from pose_tracking.utils.geom import rotate_pts_batch, transform_pts_batch
from pose_tracking.utils.misc import pick_library
from torch import nn
from torch.nn import functional as F

try:
    from pose_tracking.chamfer_distance import ChamferDistance
except Exception as e:
    print(f"Failed to import a custom ChamferDistance: {e}. Loading an alternative.")
    ChamferDistance = None


def normalize_quaternion(quat, eps=1e-8):
    return F.normalize(quat, p=2, dim=-1, eps=eps)


def geodesic_loss(pred_quat, true_quat):
    pred_quat = normalize_quaternion(pred_quat)
    true_quat = normalize_quaternion(true_quat)

    dot_product = torch.sum(pred_quat * true_quat, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angles = 2 * torch.acos(torch.abs(dot_product))
    return torch.mean(angles)


def rot_pts_displacement_loss(pred_rot_mat, true_rot_mat, pts):
    loss_rot = torch.abs(rotate_pts_batch(pred_rot_mat, pts) - rotate_pts_batch(true_rot_mat, pts))
    return torch.mean(loss_rot)


def videopose_loss(pred_quat, true_quat, eps=1e-8):
    # https://github.com/ApoorvaBeedu/VideoPose/models/loss.py
    pred_quat = normalize_quaternion(pred_quat, eps=eps)
    true_quat = normalize_quaternion(true_quat, eps=eps)
    inn_prod = torch.mm(pred_quat, true_quat.t())
    inn_prod = inn_prod.diag()

    quat_loss = 1 - (inn_prod).abs().mean()
    quat_reg_loss = (1 - pred_quat.norm(dim=1).mean()).abs()

    return quat_loss + quat_reg_loss


def geodesic_loss_mat(pred_rot, true_rot):
    R_diffs = pred_rot @ true_rot.permute(0, 2, 1)
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    dists = torch.acos(torch.clamp((traces - 1) / 2, -1, 1))
    return torch.mean(dists)


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


def kpt_cross_ratio_loss(kpts_pred, bbox_2d_kpts_collinear_idxs, cr=4 / 3):
    # LCR = Smooth`(CR^2 − ||c − a||^2||d − b||^2/||c − b||^2||d − a||^2 ),
    # where a, b, c, d are arbitrary four collinear kpts

    if len(bbox_2d_kpts_collinear_idxs.shape) > 1:
        lcr = torch.tensor(0.0, device=kpts_pred.device)
        if len(bbox_2d_kpts_collinear_idxs.shape) == 3:
            for bidx in range(bbox_2d_kpts_collinear_idxs.shape[0]):
                lcr += kpt_cross_ratio_loss(kpts_pred[bidx], bbox_2d_kpts_collinear_idxs[bidx])
        else:
            for idx_quad in bbox_2d_kpts_collinear_idxs:
                lcr += kpt_cross_ratio_loss(kpts_pred, idx_quad)
        return lcr

    a, b, c, d = kpts_pred[bbox_2d_kpts_collinear_idxs]
    cr_gt = torch.tensor(cr**2, device=kpts_pred.device)
    lcr = F.smooth_l1_loss(
        (torch.norm(c - a) ** 2 * torch.norm(d - b) ** 2) / (torch.norm(c - b) ** 2 * torch.norm(d - a) ** 2),
        cr_gt,
    )
    return lcr


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


class MixedTranslationLoss(nn.Module):
    def __init__(self):
        super(MixedTranslationLoss, self).__init__()
        self.huber = get_t_loss("huber")
        self.huber_norm = get_t_loss("huber_norm")
        self.huber_angle = get_t_loss("angle")

    def forward(self, pred, true):
        huber = self.huber(pred, true)
        huber_norm = self.huber_norm(pred, true)
        huber_angle = self.huber_angle(pred, true)
        return huber + huber_norm + huber_angle


class NormTranslationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(NormTranslationLoss, self).__init__()
        self.huber = get_t_loss("huber")
        self.eps = eps

    def forward(self, pred, true):
        pred = pred / (pred.norm(dim=-1, keepdim=True) + self.eps)
        true = true / (true.norm(dim=-1, keepdim=True) + self.eps)
        return self.huber(pred, true)


class AngleTranslationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(AngleTranslationLoss, self).__init__()
        self.huber = get_t_loss("huber")
        self.eps = eps

    def forward(self, pred, true):
        angle = torch.acos(
            torch.clamp(torch.sum(pred * true, dim=-1) / (pred.norm(dim=-1) * true.norm(dim=-1) + self.eps), -1, 1)
        )
        return self.huber(angle, torch.zeros_like(angle))


def get_t_loss(t_loss_name):
    if t_loss_name == "mse":
        criterion_trans = nn.MSELoss()
    elif t_loss_name == "mae":
        criterion_trans = nn.L1Loss()
    elif t_loss_name == "huber":
        criterion_trans = nn.SmoothL1Loss()
    elif t_loss_name == "huber_norm":
        criterion_trans = NormTranslationLoss()
    elif t_loss_name == "angle":
        criterion_trans = AngleTranslationLoss()
    elif t_loss_name == "mixed":
        criterion_trans = MixedTranslationLoss()
    else:
        raise ValueError(f"Unknown translation loss name: {t_loss_name}")
    return criterion_trans


def get_rot_loss(rot_loss_name):
    if rot_loss_name == "displacement":
        criterion_rot = rot_pts_displacement_loss
    elif rot_loss_name == "geodesic":
        criterion_rot = geodesic_loss
    elif rot_loss_name == "mse":
        criterion_rot = nn.MSELoss()
    elif rot_loss_name == "mae":
        criterion_rot = nn.L1Loss()
    elif rot_loss_name == "huber":
        criterion_rot = nn.SmoothL1Loss()
    elif rot_loss_name == "videopose":
        criterion_rot = videopose_loss
    else:
        raise ValueError(f"Unknown rotation loss name: {rot_loss_name}")
    return criterion_rot
