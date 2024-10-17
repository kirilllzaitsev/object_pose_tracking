import time

import cv2
import numpy as np
import torch
from pose_tracking.utils.common import cast_to_numpy
from skimage.measure import ransac
from skimage.transform import AffineTransform, FundamentalMatrixTransform

try:
    from lightglue.utils import load_image, rbd
except ImportError:
    print("lightglue not installed, some funcs not available")


def is_torch(x):
    return isinstance(x, torch.Tensor)


def get_good_matches_mask(mkpts0, mkpts1, thresh=0.1, min_samples=8, max_trials=1000, transform_name="fundamental"):
    use_torch = is_torch(mkpts0)
    src_pts = cast_to_numpy(mkpts0)
    dst_pts = cast_to_numpy(mkpts1)
    if transform_name == "fundamental":
        model = FundamentalMatrixTransform
    else:
        model = AffineTransform
    model, inliers = ransac(
        (src_pts, dst_pts),
        model,
        min_samples=min_samples,
        residual_threshold=thresh,
        max_trials=max_trials,
    )
    inliers = inliers.astype(bool)
    if use_torch:
        inliers = torch.from_numpy(inliers).to(mkpts0.device)
    return inliers


def get_pose_from_matches(mkpts0, mkpts1, camera_matrix, ransac_thresh=0.5, ransac_conf=0.99999):
    if len(mkpts0) < 5:
        raise ValueError("Not enough matches to estimate pose")

    mkpts0_em = cast_to_numpy(mkpts0)
    mkpts1_em = cast_to_numpy(mkpts1)
    camera_matrix_em = cast_to_numpy(camera_matrix, dtype=np.float64)

    mkpts0_em = (mkpts0_em - camera_matrix_em[[0, 1], [2, 2]][None]) / camera_matrix_em[[0, 1], [0, 1]][None]
    mkpts1_em = (mkpts1_em - camera_matrix_em[[0, 1], [2, 2]][None]) / camera_matrix_em[[0, 1], [0, 1]][None]
    ransac_thr = ransac_thresh / np.mean([camera_matrix[0, 0], camera_matrix[1, 1]])

    E, mask = cv2.findEssentialMat(
        mkpts0_em,
        mkpts1_em,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=ransac_conf,
        threshold=ransac_thr,
    )

    # inliers1 = mkpts0_em[mask.ravel() == 1]
    # inliers2 = mkpts1_em[mask.ravel() == 1]
    # _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, cameraMatrix=camera_matrix_em)

    # recover pose from E
    best_num_inliers = 0
    res = {}
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, mkpts0_em, mkpts1_em, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            res["R"] = R
            res["t"] = t[:, 0]
            best_num_inliers = n
    res["num_inliers"] = best_num_inliers
    return res


def get_matches(image0_rgb, image1_rgb, extractor, matcher):

    # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
    image0 = image0_rgb.cuda()
    image1 = image1_rgb.cuda()

    times = []
    for _ in range(1):
        start = time.time()
        # extract local features
        feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
        feats1 = extractor.extract(image1)

        # match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})
        times.append(time.time() - start)

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
    matches = matches01["matches"]  # indices with shape (K,2)
    mkpts0 = feats0["keypoints"][matches[..., 0]].cpu()  # coordinates in image #0, shape (K,2)
    mkpts1 = feats1["keypoints"][matches[..., 1]].cpu()  # coordinates in image #1, shape (K,2)

    return {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "scores": matches01["scores"],
        "times": times,
    }
