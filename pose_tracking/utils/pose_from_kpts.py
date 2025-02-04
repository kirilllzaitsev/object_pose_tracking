import numpy as np
import open3d as o3d


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
