import os
from glob import glob
from pathlib import Path

import numpy as np
from pose_tracking.config import (
    BENCHMARK_DIR,
    YCB_MESHES_DIR,
    YCBINEOAT_SCENE_DIR,
    logger,
)
from pose_tracking.dataset.ycbineoat import YCBineoatDatasetBenchmark
from pose_tracking.utils.common import print_args
from pose_tracking.utils.eval_utils import get_preds_path_benchmark
from pose_tracking.utils.io import load_color, load_pose
from pose_tracking.utils.video_utils import save_video
from pose_tracking.utils.vis import draw_poses_on_video
from tqdm import tqdm


def create_ycbvineoat_videos(ds_name, model_name, obj_names=None):

    for obj_name in tqdm(obj_names, desc="Object"):
        preds_path = get_preds_path_benchmark(model_name, obj_name, ds_name=ds_name)
        logger.info(f"Creating video for {obj_name} with model {model_name}")
        logger.info(f"{preds_path=}")
        scene_dir = f"{YCBINEOAT_SCENE_DIR}/{obj_name}"
        ds = YCBineoatDatasetBenchmark(
            preds_path=preds_path,
            video_dir=scene_dir,
            shorter_side=None,
            zfar=np.inf,
            include_rgb=True,
            include_depth=True,
            include_gt_pose=True,
            include_mask=False,
            ycb_meshes_dir=YCB_MESHES_DIR,
        )
        model = ds.mesh
        bbox = ds.mesh_bbox
        rgbs = []
        intrinsics = []
        poses_obj = []
        fps = 20
        num_seconds = 20
        for sample_idx, sample in enumerate(ds):
            rgbs.append(sample["rgb"])
            intrinsics.append(sample["intrinsics"])
            poses_obj.append(sample["pose"])
            if sample_idx == fps * num_seconds:
                break
        rgb_bbox = draw_poses_on_video(rgbs, intrinsics, poses_obj, bbox=bbox, bbox_color=(255, 255, 0), scale=0.05)

        images = rgb_bbox
        frame_height, frame_width = images[0].shape[:2]
        save_path = BENCHMARK_DIR / ds_name / f"{obj_name}_{model_name}.mp4"

        save_video(images, save_path, frame_height, frame_width, fps, live_preview=False)


def save_videos_for_obj(exp_dir, exp_name, save_dir, obj_name, intrinsics=None, bbox=None, fps=10):
    poses_pred = []
    poses_gt = []
    rgbs = []
    exp_dir = Path(exp_dir)
    preds_dir = exp_dir / "preds" / obj_name
    if intrinsics is None:
        intrinsics = np.loadtxt(preds_dir / "intrinsics.txt")
    paths = sorted(glob(str(preds_dir / "rgb" / "*.png")))
    for path in tqdm(paths, leave=True, desc="Paths"):
        rgb = load_color(path)
        filename = Path(path).stem
        pred = load_pose(preds_dir / "poses" / f"{filename}.txt")
        gt = load_pose(preds_dir / "poses_gt" / f"{filename}.txt")
        rgbs.append(rgb)
        poses_pred.append(pred)
        poses_gt.append(gt)

    rgb_bbox = draw_poses_on_video(
        rgbs,
        intrinsics,
        poses_pred=poses_pred,
        poses_gt=poses_gt,
        bbox=bbox,
        bbox_color=(255, 255, 0),
        scale=0.05,
    )
    images = rgb_bbox
    frame_height, frame_width = images[0].shape[:2]

    video_dir = f"{save_dir}/{obj_name}"
    save_path = os.path.join(video_dir, f"{exp_name}.mp4")

    save_video(images, save_path, frame_height, frame_width, fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--ds_name", type=str, required=True, choices=["ycbineoat"])
    parser.add_argument("--model_name", type=str, required=False, default="test")
    parser.add_argument("--obj_names", nargs="+", type=str, required=False)
    args, _ = parser.parse_known_args()
    print_args(args)
    create_ycbvineoat_videos(ds_name=args.ds_name, model_name=args.model_name, obj_names=args.obj_names)
