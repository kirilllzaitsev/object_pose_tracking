import time

import cv2
import numpy as np
import torch
from pose_tracking.utils.common import cast_to_numpy
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

try:
    from lightglue.utils import load_image, rbd
except ImportError:
    print("lightglue not installed, some funcs not available")


def is_torch(x):
    return isinstance(x, torch.Tensor)


def get_good_matches_mask(mkpts0, mkpts1, thresh=0.1, min_samples=8, max_trials=1000):
    use_torch = is_torch(mkpts0)
    src_pts = cast_to_numpy(mkpts0)
    dst_pts = cast_to_numpy(mkpts1)
    model, inliers = ransac(
        (src_pts, dst_pts),
        FundamentalMatrixTransform,
        min_samples=min_samples,
        residual_threshold=thresh,
        max_trials=max_trials,
    )
    inliers = inliers.astype(bool)
    if use_torch:
        inliers = torch.from_numpy(inliers).to(mkpts0.device)
    return inliers


def get_pose_from_matches(mkpts0, mkpts1, camera_matrix, ransac_thresh=1.0):
    if len(mkpts0) < 5:
        raise ValueError("Not enough matches to estimate pose")

    mkpts0_em = cast_to_numpy(mkpts0)
    mkpts1_em = cast_to_numpy(mkpts1)
    camera_matrix_em = cast_to_numpy(camera_matrix, dtype=np.float64)

    E, mask = cv2.findEssentialMat(
        mkpts0_em,
        mkpts1_em,
        cameraMatrix=camera_matrix_em,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thr,
    )

    inliers1 = mkpts0_em[mask.ravel() == 1]
    inliers2 = mkpts1_em[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, inliers1, inliers2, cameraMatrix=camera_matrix_em)
    return {"R": R, "t": t}


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
