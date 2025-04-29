import functools

import numpy as np
import torch
from pose_tracking.metrics import normalize_rotation_matrix
from pose_tracking.utils.geom import rotate_pts, rotate_pts_batch, transform_pts_batch
from pose_tracking.utils.misc import pick_library
from torch import nn
from torch.nn import functional as F


def normalize_quaternion(quat, eps=1e-8):
    return F.normalize(quat, p=2, dim=-1, eps=eps)


def geodesic_loss(pred_quat, true_quat):
    pred_quat = normalize_quaternion(pred_quat)
    true_quat = normalize_quaternion(true_quat)

    dot_product = torch.sum(pred_quat * true_quat, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angles = 2 * torch.acos(torch.abs(dot_product))
    return torch.mean(angles)


def rot_pts_displacement_loss(pred_rot_mat, true_rot_mat, pts, dist_loss="mse"):
    pred = rotate_pts_batch(pred_rot_mat, pts)
    true = rotate_pts_batch(true_rot_mat, pts)
    if dist_loss == "mae":
        loss_rot = F.l1_loss(pred, true)
    elif dist_loss == "mse":
        loss_rot = F.mse_loss(pred, true)
    elif dist_loss == "huber":
        loss_rot = F.huber_loss(pred, true)
    else:
        raise ValueError(f"Unknown distance loss: {dist_loss}")
    return loss_rot


def videopose_loss(pred_quat, true_quat, eps=1e-8):
    # https://github.com/ApoorvaBeedu/VideoPose/models/loss.py
    pred_quat_norm = normalize_quaternion(pred_quat, eps=eps)
    true_quat = normalize_quaternion(true_quat, eps=eps)
    inn_prod = torch.mm(pred_quat_norm, true_quat.t())
    inn_prod = inn_prod.diag()

    quat_loss = 1 - (inn_prod).abs().mean()
    quat_reg_loss = (1 - pred_quat.norm(dim=1).mean()).abs()

    return quat_loss + quat_reg_loss


def geodesic_loss_mat(rot_pred, rot_gt, sym_type=None, do_return_deg=False, do_reduce=True, use_eps=True, **kwargs):
    """
    rot_pred, rot_gt: torch.Tensor of shape (3, 3)
    sym_type: full/partial wrt vertical axis
    Returns: rotation error in degrees (scalar)
    """

    if isinstance(rot_pred, np.ndarray):
        rot_pred = torch.tensor(rot_pred).float()
    if isinstance(rot_gt, np.ndarray):
        rot_gt = torch.tensor(rot_gt).float()

    if rot_pred.ndim == 3:
        thetas = [
            geodesic_loss_mat(
                rot_pred[i], rot_gt[i], sym_type=sym_type, do_return_deg=do_return_deg, do_reduce=do_reduce
            )
            for i in range(rot_pred.shape[0])
        ]
        thetas = torch.stack(thetas)
        if do_reduce:
            thetas = thetas.mean()
        return thetas

    y = torch.tensor([0.0, 1.0, 0.0], device=rot_pred.device)
    eps = 1e-6 if use_eps else 0

    if sym_type == "full":
        # eg, "bottle", "can", "bowl"
        y1 = rot_pred @ y
        y2 = rot_gt @ y
        y1 = y1.flatten()
        y2 = y2.flatten()
        cos_theta = torch.clamp(torch.dot(y1, y2) / (y1.norm() * y2.norm()), -1.0 + eps, 1.0 - eps)
        theta = torch.acos(cos_theta)

    elif sym_type == "partial":
        # eg, "phone", "eggbox", "glue"
        y_180_RT = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=rot_pred.device))
        R = rot_gt @ rot_pred.transpose(-1, -2)
        R_sym = rot_gt @ y_180_RT @ rot_pred.transpose(-1, -2)

        trace_R = torch.clamp((R.trace() - 1) / 2, -1.0 + eps, 1.0 - eps)
        trace_R_sym = torch.clamp((R_sym.trace() - 1) / 2, -1.0 + eps, 1.0 - eps)

        theta = torch.min(torch.acos(trace_R), torch.acos(trace_R_sym))

    else:
        R_rel = rot_pred.transpose(-1, -2) @ rot_gt
        trace = torch.clamp((torch.einsum("...ii", R_rel) - 1) / 2, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(trace)

    if do_reduce:
        theta = theta.mean()
    if do_return_deg:
        theta = theta * 180 / torch.pi
    return theta


def compute_adds_loss(pose_pred, pose_gt, pts):
    assert len(pose_pred) == len(pose_gt), f"{len(pose_pred)=} vs {len(pose_gt)=}"
    assert len(pose_pred) > 0
    if pose_pred.shape[-2:] == (3, 3):
        transform_fn = rotate_pts
    else:
        transform_fn = transform_pts_batch
    TXO_gt_points = transform_fn(pts, pose_gt)
    TXO_pred_points = transform_fn(pts, pose_pred)
    dists_squared = (TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)) ** 2
    dists = dists_squared
    dists_norm_squared = dists_squared.sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    losses = dists_squared[ids_row, assign, ids_col].mean(dim=(-1, -2, -3))
    return losses


def compute_add_loss(pose_pred, pose_gt, pts):
    if pose_pred.shape[-2:] == (3, 3):
        transform_fn = rotate_pts
    else:
        transform_fn = transform_pts_batch
    lib = pick_library(pts)
    dists = lib.mean((transform_fn(pts, pose_gt) - transform_fn(pts, pose_pred)) ** 2)
    return dists


def compute_chamfer_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.device.type != "cpu"
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    try:
        from chamferdist.chamfer import ChamferDistance

        chamfer_dist = ChamferDistance()
        loss = 0.5 * chamfer_dist(x, y, bidirectional=True)
    except Exception as e:
        print(f"Failed to import from chamferdist: {e}. Loading an alternative.")
        try:
            from pose_tracking.chamfer_distance import ChamferDistance
        except:
            print(f"Failed to import a custom ChamferDistance: {e}. Loading an alternative.")
            return torch.tensor(torch.nan)
        chamfer_dist = ChamferDistance()
        dist1, dist2 = chamfer_dist(x, y)
        if dist1.any() == torch.inf or dist2.any() == torch.inf:
            print("WARNING: Chamfer distance is inf")
            return torch.tensor(torch.nan)
        loss = 0.5 * ((torch.mean(dist1)) + (torch.mean(dist2)))
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


def silog_loss(pred, target, eps: float = 1e-4):
    """Computes the Scale-Invariant Logarithmic (SI-Log) loss between
    prediction and target.

    Args:
        pred (Tensor): Predicted output.
        target (Tensor): Ground truth.
        weight (Optional[Tensor]): Optional weight to apply on the loss.
        eps (float): Epsilon value to avoid division and log(0).
        reduction (Union[str, None]): Specifies the reduction to apply to the
            output: 'mean', 'sum' or None.
        avg_factor (Optional[int]): Optional average factor for the loss.

    Returns:
        Tensor: The calculated SI-Log loss.
    """
    pred, target = pred.flatten(1), target.flatten(1)
    valid_mask = (target > eps).detach().float()

    diff_log = torch.log(target.clamp(min=eps)) - torch.log(pred.clamp(min=eps))

    valid_mask = (target > eps).detach() & (~torch.isnan(diff_log))
    diff_log[~valid_mask] = 0.0
    valid_mask = valid_mask.float()

    diff_log_sq_mean = (diff_log.pow(2) * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=eps)
    diff_log_mean = (diff_log * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=eps)

    loss = torch.sqrt(diff_log_sq_mean - 0.5 * diff_log_mean.pow(2))

    return loss.mean()


class SSIM(nn.Module):
    # monodepth2
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


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


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def get_t_loss(t_loss_name):
    if t_loss_name == "mse":
        criterion_trans = nn.MSELoss()
    elif t_loss_name == "rmse":
        criterion_trans = RMSELoss()
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
        criterion_rot = functools.partial(rot_pts_displacement_loss, dist_loss="mse")
    elif rot_loss_name == "geodesic":
        criterion_rot = geodesic_loss
    elif rot_loss_name == "geodesic_mat":
        criterion_rot = geodesic_loss_mat
    elif rot_loss_name == "adds":
        criterion_rot = compute_adds_loss
    elif rot_loss_name == "mse":
        criterion_rot = nn.MSELoss()
    elif rot_loss_name == "rmse":
        criterion_rot = RMSELoss()
    elif rot_loss_name == "mae":
        criterion_rot = nn.L1Loss()
    elif rot_loss_name == "huber":
        criterion_rot = nn.SmoothL1Loss()
    elif rot_loss_name == "videopose":
        criterion_rot = videopose_loss
    else:
        raise ValueError(f"Unknown rotation loss name: {rot_loss_name}")
    return criterion_rot
