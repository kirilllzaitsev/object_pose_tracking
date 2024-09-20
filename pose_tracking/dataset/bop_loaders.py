import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pose_tracking import logger
from pose_tracking.utils.io import load_json
from pose_tracking.utils.pose import combine_R_and_T
from pose_tracking.utils.trimesh_utils import load_mesh
from tqdm import tqdm


def load_list_scene(root_dir, split=None):
    if isinstance(split, str):
        if split is not None:
            split_folder = osp.join(root_dir, split)
        list_scenes = sorted(
            [
                osp.join(split_folder, scene)
                for scene in os.listdir(split_folder)
                if os.path.isdir(osp.join(split_folder, scene)) and scene != "models"
            ]
        )
    elif isinstance(split, list):
        list_scenes = []
        for scene in split:
            if not isinstance(scene, str):
                scene = f"{scene:06d}"
            if os.path.isdir(osp.join(root_dir, scene)):
                list_scenes.append(osp.join(root_dir, scene))
        list_scenes = sorted(list_scenes)
    else:
        raise NotImplementedError
    logger.info(f"Found {len(list_scenes)} scenes")
    return list_scenes


def load_scene(path, use_visible_mask=True):
    # Load rgb and mask images
    rgb_paths = sorted(Path(path).glob("rgb/*.png"))
    if use_visible_mask:
        mask_paths = sorted(Path(path).glob("mask_visib/*.png"))
    else:
        mask_paths = sorted(Path(path).glob("mask/*.png"))
    # load poses
    scene_gt = load_json(osp.join(path, "scene_gt.json"))
    scene_gt_info = load_json(osp.join(path, "scene_gt_info.json"))
    scene_camera = load_json(osp.join(path, "scene_camera.json"))
    return {
        "rgb_paths": rgb_paths,
        "mask_paths": mask_paths,
        "scene_gt": scene_gt,
        "scene_gt_info": scene_gt_info,
        "scene_camera": scene_camera,
    }


def load_metadata(root_dir, split, force_recreate: bool = False, shuffle: bool = False):
    """
    Loads metadata for the given split.
    Args:
        force_recreate: forces to recreate metadata
        shuffle: shuffle metadata rows
    """
    start_time = time.time()
    metadata = {
        "scene_id": [],
        "frame_id": [],
        "obj_id": [],
        "idx_obj": [],
        "pose": [],
        "rgb_path": [],
        "mask_path": [],
        "mask_visib_path": [],
        "depth_path": [],
        "visib_fract": [],
        "bbox_obj": [],
        "bbox_visib": [],
        "intrinsic": [],
    }
    logger.info(f"Loading metadata for split {split}")
    metadata_path = osp.join(root_dir, f"{split}_metadata.csv")
    if not osp.exists(metadata_path) or force_recreate:
        logger.info(f"Metadata at {metadata_path} will be created")
        list_scenes = load_list_scene(root_dir, split)
        for scene_path in tqdm(list_scenes, desc="Scenes"):
            scene_id = scene_path.split("/")[-1]
            rgb_paths = sorted(Path(scene_path).glob("rgb/*.png"))
            if len(rgb_paths) == 0:
                rgb_paths = sorted(Path(scene_path).glob("rgb/*.jpg"))
                assert len(rgb_paths) > 0, f"No images found in {scene_path}"

            mask_paths = sorted(Path(scene_path).glob("mask/*.png"))
            mask_paths = [str(x) for x in mask_paths]
            mask_visib_paths = sorted(Path(scene_path).glob("mask_visib/*.png"))
            mask_visib_paths = [str(x) for x in mask_visib_paths]
            depth_paths = sorted(Path(scene_path).glob("depth/*.png"))
            depth_paths = [str(x) for x in depth_paths]
            video_metadata = {}

            # load poses
            for json_name in ["scene_gt", "scene_gt_info", "scene_camera"]:
                json_path = osp.join(scene_path, json_name + ".json")
                if osp.exists(json_path):
                    video_metadata[json_name] = load_json(json_path)
                else:
                    video_metadata[json_name] = None
            # load templates metadata
            for idx_frame in tqdm(range(len(rgb_paths)), desc="Frames"):
                # get rgb path
                rgb_path = rgb_paths[idx_frame]
                # get id frame
                id_frame = int(str(rgb_path).split("/")[-1].split(".")[0])
                # get frame gt
                frame_gt = video_metadata["scene_gt"][f"{id_frame}"]
                obj_poses = np.array([combine_R_and_T(x["cam_R_m2c"], x["cam_t_m2c"]) for x in frame_gt])
                obj_ids = [int(x["obj_id"]) for x in frame_gt]

                for idx_obj in range(len(obj_ids)):
                    obj_id = obj_ids[idx_obj]
                    obj_pose = obj_poses[idx_obj]
                    mask_path = osp.join(scene_path, "mask", f"{id_frame:06d}_{idx_obj:06d}.png")
                    mask_scene_path = osp.join(scene_path, "mask", f"{id_frame:06d}.png")
                    mask_visib_path = osp.join(
                        scene_path,
                        "mask_visib",
                        f"{id_frame:06d}_{idx_obj:06d}.png",
                    )
                    depth_path = osp.join(scene_path, "depth", f"{id_frame:06d}.png")
                    if mask_path in mask_paths:
                        metadata["mask_path"].append(mask_path)
                    elif mask_scene_path in mask_paths:
                        metadata["mask_path"].append(mask_scene_path)
                    else:
                        metadata["mask_path"].append(None)
                    if mask_visib_path in mask_visib_paths:
                        metadata["mask_visib_path"].append(mask_visib_path)
                    else:
                        metadata["mask_visib_path"].append(None)
                    if depth_path in depth_paths:
                        metadata["depth_path"].append(depth_path)
                    else:
                        metadata["depth_path"].append(None)
                    metadata["scene_id"].append(scene_id)
                    metadata["frame_id"].append(id_frame)
                    metadata["obj_id"].append(obj_id)
                    metadata["idx_obj"].append(idx_obj)
                    metadata["pose"].append(obj_pose)
                    metadata["rgb_path"].append(str(rgb_path))
                    metadata["intrinsic"].append(video_metadata["scene_camera"][f"{id_frame}"]["cam_K"])
                    metadata["visib_fract"].append(
                        video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]["visib_fract"]
                        if "visib_fact" in video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]
                        else 1.0
                    )
                    metadata["bbox_obj"].append(
                        video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]["bbox_obj"]
                        if "bbox_obj" in video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]
                        else None
                    )
                    metadata["bbox_visib"].append(
                        video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]["bbox_visib"]
                        if "bbox_visib" in video_metadata["scene_gt_info"][f"{id_frame}"][idx_obj]
                        else None
                    )

        metadata = pd.DataFrame.from_dict(metadata, orient="index")
        metadata = metadata.transpose()
        metadata = metadata.sort_values(by=["scene_id", "frame_id", "obj_id"])
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if shuffle:
        metadata = metadata.sample(frac=1, random_state=1).reset_index(drop=True)
    finish_time = time.time()
    logger.info(f"Finish loading metadata of size {len(metadata)} in {finish_time - start_time:.2f} seconds")
    return metadata


def load_cad(cad_dir):
    cad_names = sorted([x for x in os.listdir(cad_dir) if x.endswith(".ply") and not x.endswith("_old.ply")])
    models_info = load_json(osp.join(cad_dir, "models_info.json"))
    cads = {}
    for cad_name in cad_names:
        cad_id = int(cad_name.split(".")[0].replace("obj_", ""))
        cad_path = osp.join(cad_dir, cad_name)
        if os.path.exists(cad_path):
            mesh = load_mesh(cad_path)
        else:
            logger.warning("CAD model unavailable")
            mesh = None
        cads[cad_id] = {
            "mesh": mesh,
            "model_info": (models_info[f"{cad_id}"] if f"{cad_id}" in models_info else models_info[cad_id]),
        }
    logger.info(f"Loaded {len(cad_names)} models for dataset done!")
    return cads
