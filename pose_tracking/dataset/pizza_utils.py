import torch
from pose_tracking.utils.geom import cam_to_2d
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from pose_tracking.utils.rotation_conversions import matrix_to_axis_angle


def get_t_2d_and_depth_from_pose(w, h, pose_gt_abs, intrinsics, do_norm=True):
    t_gt_abs = pose_gt_abs[:3]
    t_gt_2d = cam_to_2d(t_gt_abs.unsqueeze(1), intrinsics).squeeze(0)
    if do_norm:
        t_gt_2d_norm = t_gt_2d.clone()
        t_gt_2d_norm[0] = t_gt_2d_norm[0] / w
        t_gt_2d_norm[1] = t_gt_2d_norm[1] / h
    else:
        t_gt_2d_norm = t_gt_2d
    z = t_gt_abs[2].reshape(1)
    return t_gt_2d_norm, z


def extend_seq_with_pizza_args(seq):
    h, w = seq["rgb"].shape[-2:]
    delta_uv = []
    delta_rot = []
    rot_first_frame = []
    d_first_frame = []
    uv_first_frame = []
    delta_depth = []
    rot_mats = []
    device = seq["rgb"].device
    seqlen = seq["rgb"].shape[1]
    for bidx in range(seq["rgb"].shape[0]):
        rot_mats_b = convert_pose_vector_to_matrix(seq["pose"][bidx])[:, :3, :3]
        delta_uv_b = []
        delta_depth_b = []
        delta_rot_b = []
        rot_first_frame_b = []
        uv_first_frame_b = []
        d_first_frame_b = []
        for tidx in range(0, seqlen):
            pose_gt_abs_cur = seq["pose"][bidx, tidx]
            intrinsics = seq["intrinsics"][bidx, tidx]
            t_2d_cur, z_cur = get_t_2d_and_depth_from_pose(w, h, pose_gt_abs_cur, intrinsics)
            rot_mat_cur = rot_mats_b[tidx]

            if tidx == 0:
                continue

            if tidx == 0:
                delta_uv_b.append(torch.zeros(2, device=device))
                delta_depth_b.append(torch.zeros(1, device=device))
                delta_rot_b.append(torch.zeros(3, 3, device=device))
            else:
                pose_gt_abs_prev = seq["pose"][bidx, tidx - 1]
                intrinsics_prev = seq["intrinsics"][bidx, tidx - 1]
                t_2d_prev, z_prev = get_t_2d_and_depth_from_pose(w, h, pose_gt_abs_prev, intrinsics_prev)
                rot_mat_prev = rot_mats_b[tidx - 1]
                rot_delta = rot_mat_prev.T @ rot_mat_cur
                delta_uv_b.append(t_2d_cur - t_2d_prev)
                delta_depth_b.append(z_cur / z_prev - 1)
                delta_rot_b.append(rot_delta)

            if tidx == 1:
                rot_first_frame_b.append(rot_mat_cur)
                uv_first_frame_b.append(t_2d_cur)
                d_first_frame_b.append(z_cur)
        delta_uv.append(torch.stack(delta_uv_b))
        delta_depth.append(torch.stack(delta_depth_b))
        rot_first_frame.append(torch.stack(rot_first_frame_b)[0])
        d_first_frame.append(torch.stack(d_first_frame_b)[0])
        delta_rot.append(torch.stack(delta_rot_b))
        rot_mats.append(rot_mats_b)
        uv_first_frame.append(torch.stack(uv_first_frame_b)[0])

    delta_uv = torch.stack(delta_uv).to(device)
    delta_depth = torch.stack(delta_depth).to(device)
    rot_first_frame = torch.stack(rot_first_frame).to(device)
    d_first_frame = torch.stack(d_first_frame).to(device)
    uv_first_frame = torch.stack(uv_first_frame).to(device)
    delta_rot = torch.stack(delta_rot).to(device)
    rot_mats = torch.stack(rot_mats).to(device)
    seq["delta_uv"] = delta_uv
    seq["delta_depth"] = delta_depth
    seq["depth_first_frame"] = d_first_frame
    seq["uv_first_frame"] = uv_first_frame
    seq["rotation_first_frame"] = rot_first_frame
    seq["gt_delta_rotation"] = matrix_to_axis_angle(delta_rot)
    seq["gt_rotations"] = rot_mats
    seq["ratio"] = torch.ones(seq["rgb"].shape[0], 1, device=device)
    return seq
