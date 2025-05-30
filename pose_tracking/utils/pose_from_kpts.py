import numpy as np
import open3d as o3d
from pose_tracking.utils.common import cast_to_numpy
from pose_tracking.utils.geom import backproj_2d_pts
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from tqdm import tqdm


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def est_pose_ransac_feature_matching(pts_prev, pts_cur):
    # https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts_prev)
    pcd2.points = o3d.utility.Vector3dVector(pts_cur)
    pcd1_down, pcd1_fpfh = preprocess_point_cloud(pcd1, voxel_size=0.01)
    pcd2_down, pcd2_fpfh = preprocess_point_cloud(pcd2, voxel_size=0.01)
    distance_threshold = 0.01
    reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1_down,
        pcd2_down,
        pcd1_fpfh,
        pcd2_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    icp_proposal = reg_ransac.transformation
    return icp_proposal


def est_poses_from_tracks(pred_tracks, pred_visibility, ds, stop_idx=None):

    poses_pred = [cast_to_numpy(ds[0]["pose"])]

    x_prev = ds[0]
    stop_idx = stop_idx or len(ds)
    stop_idx = min(stop_idx, len(ds))
    if len(pred_tracks.shape) > 3:
        pred_tracks = pred_tracks[0]
        pred_visibility = pred_visibility[0]
    for j in tqdm(range(1, stop_idx)):
        x_cur = ds[j]
        new_pose = est_rel_pose_from_kpts(
            x_prev=x_prev,
            x_cur=x_cur,
            ts=j,
            kpt_tracks=pred_tracks,
            kpt_visibilities=pred_visibility,
            prev_pose=poses_pred[-1],
        )
        # import roma
        # from pose_tracking.dataset.ds_common import from_numpy

        # R_predicted, t_predicted = roma.rigid_points_registration(
        #     from_numpy(pts_prev_backproj), from_numpy(pts_cur_backproj)
        # )
        # new_pose[:3, :3] = R_predicted
        # new_pose[:3, 3] = t_predicted.squeeze()
        # new_pose = new_pose @ prev_pose

        poses_pred.append(new_pose)

        x_prev = x_cur
        # break
    poses_pred = np.array(poses_pred)
    return poses_pred


def est_rel_pose_from_kpts(x_prev, x_cur, ts, kpt_tracks, kpt_visibilities, prev_pose=None):
    # prev_pose = cast_to_numpy(x_prev["pose"])
    prev_pose = cast_to_numpy(prev_pose)

    pts_prev_visib = kpt_visibilities[ts - 1].squeeze(0).bool()
    pts_cur_visib = kpt_visibilities[ts].squeeze(0).bool()
    pts_prev = (kpt_tracks[ts - 1][pts_prev_visib]).cpu()
    pts_cur = (kpt_tracks[ts][pts_cur_visib]).cpu()

    prev_mask = x_prev["mask"].squeeze(0)
    cur_mask = x_cur["mask"].squeeze(0)
    # prev_mask = mask_morph(prev_mask, kernel_size=11)
    # cur_mask = mask_morph(cur_mask, kernel_size=11)

    z_prev = x_prev["depth"].clone().squeeze(0)
    z_cur = x_cur["depth"].clone().squeeze(0)
    z_prev[prev_mask == 0] = 0
    z_cur[cur_mask == 0] = 0
    z_prev = z_prev[pts_prev.long()[:, 1], pts_prev.long()[:, 0]]
    z_cur = z_cur[pts_cur.long()[:, 1], pts_cur.long()[:, 0]]
    z_prev = cast_to_numpy(z_prev).reshape(-1, 1)
    z_cur = cast_to_numpy(z_cur).reshape(-1, 1)

    pts_prev = cast_to_numpy(pts_prev)
    pts_cur = cast_to_numpy(pts_cur)
    pts_prev_backproj = backproj_2d_pts(pts_prev, K=x_prev["intrinsics"], depth=z_prev)
    pts_cur_backproj = backproj_2d_pts(pts_cur, K=x_cur["intrinsics"], depth=z_cur)

    # valid_depth_mask = ((z_prev > 0) & (z_cur > 0)).squeeze()
    # print(f"{valid_depth_mask.sum()=}")
    # pts_prev_backproj = pts_prev_backproj[valid_depth_mask]
    # pts_cur_backproj = pts_cur_backproj[valid_depth_mask]
    # pts_prev = pts_prev[valid_depth_mask]
    # pts_cur = pts_cur[valid_depth_mask]

    rel_pose = est_pose_ransac_feature_matching(pts_prev_backproj, pts_cur_backproj)
    new_pose = rel_pose @ prev_pose

    # pcd1 = o3d.geometry.PointCloud()
    # pcd2 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(pts_prev_backproj)
    # pcd2.points = o3d.utility.Vector3dVector(pts_cur_backproj)
    # # pcd1.points = o3d.utility.Vector3dVector(pts_prev_backproj_full)
    # # pcd2.points = o3d.utility.Vector3dVector(pts_cur_backproj_full)
    # threshold = 0.01
    # icp_proposal = np.eye(4)
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pcd1,
    #     pcd2,
    #     threshold,
    #     icp_proposal,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(
    #         # max_iteration=200,
    #         # relative_fitness=1e-8,
    #         # relative_rmse=1e-8,
    #     ),
    # )
    # new_pose = reg_p2p.transformation @ prev_pose
    return new_pose
