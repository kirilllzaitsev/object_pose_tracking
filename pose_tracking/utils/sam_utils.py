from pathlib import Path

import cv2
import numpy as np
from pose_tracking.utils.common import cast_to_torch
import torch
from pose_tracking.config import WORKSPACE_DIR
from pose_tracking.utils.vis import adjust_img_for_plt
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def get_sam_predictor(use_video=False, model_size="large"):
    sam_base = Path(f"{WORKSPACE_DIR}/related_work/data/segment-anything-2")
    if model_size == "large":
        checkpoint = sam_base / "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
    else:
        checkpoint = sam_base / "./checkpoints/sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
    if use_video:
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    else:
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    return predictor


def get_sam_pred(
    predictor, frame, input_point=None, input_label=None, mask_input=None, bbox=None, multimask_output=True
):
    assert (
        input_point is not None and input_label is not None
    ) or mask_input is not None, "Either input_point and input_label or mask_input must be provided"

    if isinstance(frame, torch.Tensor):
        frame = adjust_img_for_plt(frame)

    if mask_input is not None:
        mask_input = mask_input.squeeze()
        if mask_input.shape != [256, 256]:
            mask_input = cv2.resize(mask_input, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_input = mask_input[None]

    predictor.set_image(frame)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=bbox,
        mask_input=mask_input,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    return {"masks": masks, "scores": scores, "logits": logits}


def get_obj_mask_from_kpts(frame, kpts, sam_model, max_kpts=20):
    if len(kpts) > max_kpts:
        kpts = kpts[np.random.choice(len(kpts), max_kpts, replace=False)]
    input_point = cast_to_torch(kpts, include_top_list=True)
    input_label = torch.ones(input_point.shape[0])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = get_sam_pred(sam_model, frame, input_point=input_point, input_label=input_label)
    masks = pred["masks"]
    scores = pred["scores"]
    mask = masks[0]
    # show_masks(frame, masks, scores, point_coords=input_point, input_labels=input_label)
    return {"mask": mask, "scores": scores}
