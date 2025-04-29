from collections import defaultdict

import torch
import torch.nn.functional as F
from pose_tracking.utils.geom import (
    backproj_2d_pts,
    cam_to_2d,
    get_inv_pose,
    transform_pts,
    transform_pts_batch,
)

try:
    from pytorch3d.transforms.so3 import _so3_exp_map, hat, so3_log_map
except ImportError:
    print("pytorch3d not installed. some functions will not work.")
from torch import optim
from tqdm.auto import tqdm


def convert_pose_to_pytorch3d(pose):
    if pose.ndim == 2:
        pose = pose[None]
    if torch.allclose(pose[..., 3, :3], torch.zeros(3).to(pose.dtype)):
        pose = pose.clone()
        pose[..., 3, :3] = pose[..., :3, 3]
        pose[..., :3, 3] = torch.zeros(3).to(pose.dtype)
    return pose


def convert_pose_from_pytorch3d(pose):
    if pose.shape[0] == 1:
        pose = pose[0]
    if torch.allclose(pose[..., :3, 3], torch.zeros(3).to(pose.dtype)):
        pose = pose.clone()
        pose[..., :3, 3] = pose[..., 3, :3]
        pose[..., 3, :3] = torch.zeros(3).to(pose.dtype)
    return pose


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    add_batch = False
    if log_transform.ndim == 1:
        add_batch = True
        log_transform = log_transform[None]
    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(N, 4, 4, dtype=log_transform.dtype, device=log_transform.device)

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    transform = transform.permute(0, 2, 1)

    res = convert_pose_from_pytorch3d(transform)
    return res


def se3_log_map(transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [log_translation | log_rotation]`
    is done as follows:
        ```
        log_transform = log(transform)
        log_translation = log_transform[3, :3]
        log_rotation = inv_hat(log_transform[:3, :3])
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].

    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_exp_map(se3_log_map(transform)) == transform
    ```

    The conversion has a singularity around `(transform=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid division by zero in the singular case.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
            The non-finite outputs can be caused by passing small rotation angles
            to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.

    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    add_batch = False
    if transform.ndim == 2:
        add_batch = True
        transform = transform[None]
    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, :3, 3], torch.zeros_like(transform[:, :3, 3])):
        # raise ValueError("All elements of `transform[:, :3, 3]` should be 0.")
        transform = convert_pose_to_pytorch3d(transform)

    # log_rot is just so3_log_map of the upper left 3x3 block
    R = transform[:, :3, :3].permute(0, 2, 1)
    log_rotation = so3_log_map(R, eps=eps, cos_bound=cos_bound)

    # log_translation is V^-1 @ T
    T = transform[:, 3, :3]
    V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
    log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]

    res = torch.cat((log_translation, log_rotation), dim=1)
    if add_batch:
        res = res[0]
    return res


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[:, None, None]
        )
    )

    return V


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles


def energy_f(
    pts_i,
    pts_j,
    z_i,
    z_j,
    K,
    pose_i,
    pose_j,
    robust_delta=0.005,
):
    is_batch = len(pts_i.shape) == 3
    if is_batch:
        transform_pts_func = transform_pts_batch
    else:
        transform_pts_func = transform_pts
    backproj_2d_pts_func = backproj_2d_pts
    pts_i_3d = backproj_2d_pts_func(pts_i, depth=z_i, K=K)
    pts_j_3d = backproj_2d_pts_func(pts_j, depth=z_j, K=K)
    pose_i_inv = get_inv_pose(pose_i)
    pose_j_inv = get_inv_pose(pose_j)
    pts_i_3d_obj = transform_pts_func(pts=pts_i_3d, pose=pose_i_inv)
    pts_j_3d_obj = transform_pts_func(pts=pts_j_3d, pose=pose_j_inv)
    r = pts_j_3d_obj - pts_i_3d_obj
    return F.huber_loss(r, torch.zeros_like(r), delta=robust_delta)


def energy_g(pts_i, z_i, z_j, K, pose_i, pose_j, normals_i, robust_delta=0.005):
    is_batch = len(pts_i.shape) == 3
    if is_batch:
        transform_pts_func = transform_pts_batch
    else:
        transform_pts_func = transform_pts
    backproj_2d_pts_func = backproj_2d_pts
    pts_i_3d = backproj_2d_pts_func(pts_i, depth=z_i, K=K)
    pose_i_inv = get_inv_pose(pose_i)
    pts_obj = transform_pts_func(pts_i_3d, pose=pose_i_inv)
    pts_j = transform_pts_func(pts_obj, pose=pose_j)
    pts_j_2d = cam_to_2d(pts_j, K)
    pts_j_3d = backproj_2d_pts_func(pts_j_2d, depth=z_j, K=K)
    pose_j_inv = get_inv_pose(pose_j)
    pts_obj_3d = transform_pts_func(pts_j_3d, pose=pose_j_inv)
    pts_i_3d_2 = transform_pts_func(pts_obj_3d, pose=pose_i)
    diff = pts_i_3d_2 - pts_i_3d
    if is_batch:
        normals_i_pts = []
        bs = pts_i.shape[0]
        for i in range(bs):
            normals_i_pts.append(normals_i[i, pts_i[i, :, 1].long(), pts_i[i, :, 0].long()])
        normals_i_pts = torch.stack(normals_i_pts, dim=0).to(pts_i.device)
    else:
        normals_i_pts = normals_i[..., pts_i[..., 1].long(), pts_i[..., 0].long(), :]

    return F.huber_loss(diff * normals_i_pts, torch.zeros_like(diff), delta=robust_delta)


class Frame:
    def __init__(self, pose, surface_normals, pts_2d, depth, rgb, K):
        self.pose = pose
        self.surface_normals = surface_normals
        self.pts_2d = pts_2d
        self.rgb = rgb
        self.depth = depth
        self.K = K

        self.z = depth[pts_2d[:, 1].long(), pts_2d[:, 0].long()]


def compute_loss(frames: list[Frame], pose_vec_to_matrix_fn):
    total_loss = 0.0
    for i in range(len(frames)):
        frame_i = frames[i]
        for j in range(len(frames)):
            if i == j:
                continue
            frame_j = frames[j]
            pose_i = pose_vec_to_matrix_fn(frame_i.pose)
            pose_j = pose_vec_to_matrix_fn(frame_j.pose)
            loss_f = energy_f(
                pts_i=frame_i.pts_2d,
                pts_j=frame_j.pts_2d,
                z_i=frame_i.z,
                z_j=frame_j.z,
                K=frame_i.K,
                pose_i=pose_i,
                pose_j=pose_j,
            )
            total_loss += loss_f
            loss_g = energy_g(
                pts_i=frame_i.pts_2d,
                z_i=frame_i.z,
                z_j=frame_j.z,
                K=frame_i.K,
                pose_i=pose_i,
                pose_j=pose_j,
                normals_i=frame_i.surface_normals,
            )
            total_loss += loss_g
    return total_loss


def compute_loss_v2(frames: list[Frame], pose_vec_to_matrix_fn):
    total_loss = 0.0
    n = len(frames)
    idxs = [(i, j) for i in range(n) for j in range(n) if i != j and i < j]
    pose_is = []
    pose_js = []
    pts_is = []
    pts_js = []
    z_is = []
    z_js = []
    Ks = []
    normals_is = []
    for i, j in idxs:
        frame_i = frames[i]
        frame_j = frames[j]
        pose_is.append(pose_vec_to_matrix_fn(frame_i.pose))
        pose_js.append(pose_vec_to_matrix_fn(frame_j.pose))
        pts_is.append(frame_i.pts_2d)
        pts_js.append(frame_j.pts_2d)
        z_is.append(frame_i.z)
        z_js.append(frame_j.z)
        Ks.append(frame_i.K)
        normals_is.append(frame_i.surface_normals)

    device = "cpu"
    device = "cuda"
    pose_i = torch.stack(pose_is).to(device)
    pose_j = torch.stack(pose_js).to(device)
    pts_i = torch.stack(pts_is).to(device)
    pts_j = torch.stack(pts_js).to(device)
    z_i = torch.stack(z_is).to(device)
    z_j = torch.stack(z_js).to(device)
    K = torch.stack(Ks).to(device)
    normals_i = torch.stack(normals_is).to(device)

    loss_f = energy_f(
        pts_i=pts_i,
        pts_j=pts_j,
        z_i=z_i,
        z_j=z_j,
        K=K,
        pose_i=pose_i,
        pose_j=pose_j,
    )
    total_loss += loss_f
    # loss_g = energy_g(
    #     pts_i=pts_i,
    #     z_i=z_i,
    #     z_j=z_j,
    #     K=K,
    #     pose_i=pose_i,
    #     pose_j=pose_j,
    #     normals_i=normals_i,
    # )
    # total_loss += loss_g
    return total_loss


def optimize(poses, frames, pose_vec_to_matrix_fn, lr=1e-3, num_iterations=100):
    optimizer = optim.Adam(poses, lr=lr)
    losses = []
    pbar = tqdm(range(num_iterations), total=num_iterations, leave=False, disable=False)
    grad_norms = defaultdict(list)
    total_grad_norms = []
    for it in pbar:
        optimizer.zero_grad()
        loss = compute_loss_v2(frames, pose_vec_to_matrix_fn=pose_vec_to_matrix_fn)
        loss.backward()
        # print grads
        total_grad_norm = 0.0
        for pidx, p in enumerate(poses):
            if p.grad is None:
                continue
            grad_norm = p.grad.norm().item()
            total_grad_norm += grad_norm
            grad_norms[pidx].append(grad_norm)
        # print(f"{it=} | grad norm: {total_grad_norm}")
        total_grad_norms.append(total_grad_norm)
        optimizer.step()
        # print(f"Iteration {it}: Loss = {loss.item()}")
        loss_val = loss.item()
        losses.append(loss_val)
        pbar.set_postfix({"loss": loss_val})
    return {"losses": losses, "grad_norms": grad_norms, "total_grad_norms": total_grad_norms}
