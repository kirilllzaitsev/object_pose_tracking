import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from albumentations import transforms as A
from pose_tracking.config import RELATED_DIR
from pose_tracking.utils.common import cast_to_numpy
from skimage.measure import ransac
from skimage.transform import AffineTransform, FundamentalMatrixTransform

try:
    from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
except ImportError:
    print("lightglue not installed, some funcs not available")

try:
    from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
except ImportError:
    print("cotracker not installed, some funcs not available")


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


def get_pose_from_matches(mkpts0, mkpts1, camera_matrix, ransac_thresh=1.0, ransac_conf=0.99999):
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
        prob=ransac_conf,
        threshold=ransac_thresh,
    )

    inliers1 = mkpts0_em[mask.ravel() == 1]
    inliers2 = mkpts1_em[mask.ravel() == 1]

    best_num_inliers = -1
    res = {}
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, inliers1, inliers2, cameraMatrix=camera_matrix_em)
        if n > best_num_inliers:
            res["R"] = R
            res["t"] = t[:, 0]
            best_num_inliers = n
    res["num_inliers"] = best_num_inliers
    return res


def get_pose_from_3d_2d_matches(kpts_3d, kpts_2d, intrinsics, ransac_conf=0.99999):
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        cast_to_numpy(kpts_3d),
        cast_to_numpy(kpts_2d),
        cast_to_numpy(intrinsics),
        None,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=500,
        confidence=ransac_conf,
    )
    if not success:
        print("Pose estimation failed.")
        rot_mat_pred_bidx = np.eye(3)
    else:
        rot_mat_pred_bidx, _ = cv2.Rodrigues(rvec)
    return {
        "R": rot_mat_pred_bidx,
        "t": tvec,
        "num_inliers": len(inliers) if inliers is not None else 0,
    }


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


def get_matches_loftr(image0_rgb, image1_rgb, matcher, mask0=None, mask1=None, use_native_masking=False):

    to_gray = A.ToGray(num_output_channels=1, p=1.0)
    image0 = torch.from_numpy(
        to_gray(image=image0_rgb.permute(1, 2, 0).numpy())["image"].transpose(2, 0, 1)[None]
    ).cuda()
    image1 = torch.from_numpy(
        to_gray(image=image1_rgb.permute(1, 2, 0).numpy())["image"].transpose(2, 0, 1)[None]
    ).cuda()

    batch = {
        "image0": image0,
        "image1": image1,
    }
    if use_native_masking:
        assert mask0 is not None and mask1 is not None
        batch_masks = {
            "mask0": torch.from_numpy(mask0[None]).cuda(),
            "mask1": torch.from_numpy(mask1[None]).cuda(),
        }
        batch.update(batch_masks)
    times = []
    with torch.no_grad():
        for _ in range(1):
            start = time.time()
            matcher(batch)  # batch = {'image0': img0, 'image1': img1}
            times.append(time.time() - start)
        mkpts0 = batch["mkpts0_f"].cpu()
        mkpts1 = batch["mkpts1_f"].cpu()
    return {
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "scores": batch["mconf"].cpu(),
        "times": times,
    }


def load_kpt_det_and_match_loftr(ckpt_filename="indoor_ds_new", use_quadattn=False):
    ckpt_filename = Path(ckpt_filename).stem
    if use_quadattn:
        from quadattn.src.loftr import LoFTR

        cfg = yaml.load(
            open(f"{RELATED_DIR}/kpt_det_match/QuadTreeAttention/FeatureMatching/quadattn/loftr_indoor.yaml"),
            Loader=yaml.FullLoader,
        )
        matcher = LoFTR(cfg)
        ckpt_path = f"{RELATED_DIR}/kpt_det_match/QuadTreeAttention/FeatureMatching/quadattn/{ckpt_filename}.ckpt"
    else:
        from loftr.src.loftr import LoFTR, default_cfg

        matcher = LoFTR(config=default_cfg)
        ckpt_path = f"{RELATED_DIR}/kpt_det_match/LoFTR/weights/{ckpt_filename}.ckpt"
    matcher.load_state_dict(torch.load(ckpt_path)["state_dict"])
    matcher = matcher.eval().cuda()
    for p in matcher.parameters():
        p.requires_grad = False
    return matcher


def load_kpt_det_and_match(features, filter_threshold=0.1):
    # TODO: provide configs for extractor/matcher
    extractor = load_extractor(features)

    matcher = LightGlue(features=features, filter_threshold=filter_threshold).eval().cuda()  # load the matcher

    for p in extractor.parameters():
        p.requires_grad = False
    for p in matcher.parameters():
        p.requires_grad = False

    return extractor, matcher


def load_extractor(features, max_num_keypoints=1024):
    if features == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()  # load the extractor
        for n, p in extractor.named_parameters():
            if "convPa" in n or "convPb" in n:
                p.requires_grad = False
    elif features == "disk":
        # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
        extractor = DISK(max_num_keypoints=max_num_keypoints).eval()  # load the extractor
    elif features == "sift":
        extractor = SIFT(max_num_keypoints=max_num_keypoints).eval()  # load the extractor
    elif features == "aliked":
        extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval()
    elif features == "doghardnet":
        extractor = DoGHardNet(max_num_keypoints=max_num_keypoints).eval()
    else:
        raise ValueError(features)
    return extractor


def load_tracker(use_stream_tracker=True, use_online_tracker=True, use_v2=False, stream_window_len=16):
    if use_stream_tracker:
        cotracker = CoTrackerOnlinePredictor(
            checkpoint=os.path.join(
                f"{RELATED_DIR}/kpt_tracking/co-tracker",
                "./checkpoints/scaled_online.pth",
            ),
            window_len=stream_window_len,
            v2=use_v2,
        )
    else:
        ckpt_filename = "scaled_online" if use_online_tracker else "scaled_offline"
        window_len = 16 if use_online_tracker else 60
        cotracker = CoTrackerPredictor(
            checkpoint=os.path.join(
                f"{RELATED_DIR}/kpt_tracking/co-tracker",
                f"./checkpoints/{ckpt_filename}.pth",
            ),
            v2=use_v2,
            offline=not use_online_tracker,
            window_len=window_len,
        )
    return cotracker


def get_kpt_within_mask_indicator(keypoints, binary_mask):
    if not isinstance(keypoints, torch.Tensor):
        keypoints = torch.tensor(keypoints)
    keypoints = keypoints.long().to(binary_mask.device)
    x, y = keypoints[..., 0], keypoints[..., 1]
    # inside_bounds = (0 <= y) & (y < binary_mask.shape[0]) & (0 <= x) & (x < binary_mask.shape[1])
    indicator = (binary_mask[..., y, x] == 1).long()
    return indicator


def extract_kpts(x, extractor, do_normalize=False, use_zeros_for_pad=True):
    bs, c, h, w = x.shape
    memory_key_padding_mask = None
    if bs > 1:
        extracted_kpts = [extractor.extract(x[i : i + 1]) for i in range(bs)]
        extracted_kpts = [{k: v[0] for k, v in kpts.items()} for kpts in extracted_kpts]
        # pad with zeros up to the max number of keypoints
        max_kpts = max([len(kpts["keypoints"]) for kpts in extracted_kpts])
        memory_key_padding_mask = torch.zeros(bs, max_kpts, dtype=torch.bool, device=x.device)
        # extracted_kpts = copy.deepcopy(extracted_kpts)
        for bidx, kpts in enumerate(extracted_kpts):
            for k in ["keypoints", "descriptors"]:
                pad_len = max_kpts - len(kpts[k])
                if pad_len > 0:
                    if use_zeros_for_pad:
                        memory_key_padding_mask[bidx, -pad_len:] = True
                        kpts[k] = F.pad(kpts[k], (0, 0, 0, pad_len), value=0)
                    else:
                        # duplicate random pts
                        pad_idxs = torch.randint(0, len(kpts[k]), (pad_len,))
                        kpts[k] = torch.cat([kpts[k], kpts[k][pad_idxs]])
        extracted_kpts = {k: torch.stack([v[k] for v in extracted_kpts], dim=0) for k in ["keypoints", "descriptors"]}
    else:
        extracted_kpts = extractor.extract(x)

    if do_normalize:
        extracted_kpts["keypoints"] = extracted_kpts["keypoints"] / torch.tensor(
            [w, h], dtype=extracted_kpts["keypoints"].dtype
        ).to(extracted_kpts["keypoints"].device)

    return {**extracted_kpts, "padding_mask": memory_key_padding_mask}


def kabsch_torch(P, Q):
    """
    https://hunterheidenreich.com/posts/kabsch_algorithm/
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, t, rmsd


def kabsch_torch_batched(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        flip_expanded = flip.unsqueeze(1).expand_as(Vt[:, -1])
        new_last = torch.where(flip_expanded, -Vt[:, -1], Vt[:, -1])
        Vt = torch.cat([Vt[:, :-1], new_last.unsqueeze(1)], dim=1)

    # Optimal rotation
    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1])

    return R, t, rmsd
