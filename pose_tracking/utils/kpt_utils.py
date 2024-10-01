import cv2
import numpy as np
from pose_tracking.utils.common import cast_to_numpy
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def get_good_matches_mask(mkpts0, mkpts1, thresh=0.1, min_samples=8, max_trials=1000):
    src_pts = mkpts0
    dst_pts = mkpts1
    model, inliers = ransac(
        (src_pts, dst_pts),
        FundamentalMatrixTransform,
        min_samples=min_samples,
        residual_threshold=thresh,
        max_trials=max_trials,
    )
    inliers = inliers.astype(bool)
    return inliers


def get_pose_from_matches(mkpts0, mkpts1, camera_matrix):
    mkpts0_em = cast_to_numpy(mkpts0)
    mkpts1_em = cast_to_numpy(mkpts1)
    camera_matrix_em = cast_to_numpy(camera_matrix, dtype=np.float64)

    E, mask = cv2.findEssentialMat(
        mkpts0_em,
        mkpts1_em,
        cameraMatrix=camera_matrix_em,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )

    inliers1 = mkpts0_em[mask.ravel() == 1]
    inliers2 = mkpts1_em[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, cameraMatrix=camera_matrix_em)
    return {"R": R, "t": t}
