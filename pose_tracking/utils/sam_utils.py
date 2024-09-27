from pathlib import Path

import cv2
import numpy as np
from pose_tracking.config import WORKSPACE_DIR
import torch
from pose_tracking.utils.vis import adjust_img_for_plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def get_sam_predictor():
    sam_base = Path(f"{WORKSPACE_DIR}/segment-anything-2")
    checkpoint = sam_base / "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
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
