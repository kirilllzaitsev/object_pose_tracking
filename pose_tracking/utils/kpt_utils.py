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
