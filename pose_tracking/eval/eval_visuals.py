import json
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
from pose_tracking.utils.vis import draw_poses_on_video, vis_bbox_2d
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
            include_pose=True,
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


def load_preds(preds_dir, include_det=False):
    poses_pred = []
    poses_gt = []
    rgbs = []
    boxes = []
    labels = []

    preds_dir = Path(preds_dir)
    intrinsics_path = preds_dir / "intrinsics.txt"
    if intrinsics_path.exists():
        intrinsics = np.loadtxt(intrinsics_path)
    else:
        intrinsics = None
    paths = sorted(glob(str(preds_dir / "rgb" / "*.png")))
    for path in tqdm(paths, leave=True, desc="Paths"):
        rgb = load_color(path)
        filename = Path(path).stem
        pred = load_pose(preds_dir / "poses" / f"{filename}.txt")
        gt = load_pose(preds_dir / "poses_gt" / f"{filename}.txt")
        rgbs.append(rgb)
        poses_pred.append(pred)
        poses_gt.append(gt)
        if include_det:
            det_path = preds_dir / "bbox" / f"{filename}.json"
            if det_path.exists():
                with open(det_path, "r") as f:
                    det = json.load(f)
                    boxes.append(det["bbox"])
                    labels.append(det["labels"])
    res = {
        "rgbs": rgbs,
        "poses_pred": poses_pred,
        "poses_gt": poses_gt,
        "intrinsics": intrinsics,
        "boxes": boxes,
        "labels": labels,
    }
    return res


def save_videos_for_obj(preds_dir, video_save_path=None, intrinsics=None, bbox=None, fps=10, include_det=False):

    res = load_preds(preds_dir, include_det=include_det)
    rgbs = res["rgbs"]
    poses_pred = res["poses_pred"]
    poses_gt = res["poses_gt"]
    intrinsics = res["intrinsics"] if intrinsics is None else intrinsics
    boxes = res["boxes"]
    labels = res["labels"]

    rgb_bbox = draw_poses_on_video(
        rgbs,
        intrinsics,
        poses_pred=poses_pred,
        poses_gt=poses_gt,
        bbox=bbox,
        bbox_color=(255, 255, 0),
        scale=0.05,
    )
    frame_height, frame_width = rgb_bbox[0].shape[:2]

    if video_save_path is None:
        video_save_path = preds_dir / "video.mp4"
    save_video(rgb_bbox, video_save_path, frame_height, frame_width, fps)

    if include_det:
        if len(boxes) == 0:
            logger.error(f"No detections found in {preds_dir}")
        else:
            imgs_det = []
            for idx in range(len(boxes)):
                img = rgbs[idx].copy()

                boxes_t = boxes[idx]
                for obj_idx in range(len(boxes_t)):
                    box = boxes_t[obj_idx]
                    img = vis_bbox_2d(
                        img,
                        box,
                        is_normalized=False,
                        label=labels[idx][obj_idx],
                        color=(255, 255, 0),
                    )
                imgs_det.append(img)
            save_video(imgs_det, preds_dir / "video_det.mp4", frame_height, frame_width, fps)


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
