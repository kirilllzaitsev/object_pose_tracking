from pose_tracking.utils.misc import is_tensor
import torch
import torch.nn.functional as F
from pose_tracking.models.matcher import box_cxcywh_to_xyxy
from torchvision.ops import roi_align
from torchvision.ops.boxes import batched_nms


def postprocess_detr_outputs(outputs, target_sizes, is_focal_loss=True):
    """https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py
    Args:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs["pred_logits"], outputs.get("pred_boxes", outputs.get("pred_bboxes"))

    # assert len(out_logits) == len(target_sizes)
    assert target_sizes.ndim == 1 or target_sizes.shape[1] == 2

    if is_focal_loss:
        prob = out_logits.sigmoid()
        scores, labels = prob.max(-1)
        scores_no_object = 1 - scores
    else:
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        scores_no_object = prob[..., -1]

    if out_bbox is None:
        boxes = torch.zeros((out_logits.shape[0], out_logits.shape[1], 4), device=out_logits.device)
    else:
        boxes = postprocess_detr_boxes(out_bbox, target_sizes)

    use_pose = "rot" in outputs and "t" in outputs
    results = []
    for idx in range(out_logits.shape[0]):
        res = {
            "scores": scores[idx],
            "labels": labels[idx],
            "boxes": boxes[idx],
            "scores_no_object": scores_no_object[idx],
        }
        if use_pose:
            res["rot"] = outputs["rot"][idx]
            res["t"] = outputs["t"][idx]
            if "center_depth" in outputs:
                res["center_depth"] = outputs["center_depth"][idx]

        results.append(res)

    return results


def postprocess_detr_boxes(out_bbox, target_sizes):
    boxes = box_cxcywh_to_xyxy(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    return boxes


def postprocess_bbox(bbox, hw=None, format="cxcywh", is_normalized=True):
    if is_normalized:
        assert hw is not None

    res = bbox.clone()
    if format == "cxcywh":
        res = box_cxcywh_to_xyxy(res)
    elif format == "xyxy":
        pass
    else:
        raise ValueError(f"Unknown format {format}")

    if is_normalized:
        res[..., [0, 2]] *= hw[1]
        res[..., [1, 3]] *= hw[0]

    return res


def postprocess_detr_outputs_nms(outputs, target_sizes):
    """https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py
    Args:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """
    out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
    bs, n_queries, n_cls = out_logits.shape

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()

    all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
    all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
    all_boxes = all_indexes // out_logits.shape[2]
    all_labels = all_indexes % out_logits.shape[2]

    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for b in range(bs):
        box = boxes[b]
        score = all_scores[b]
        lbls = all_labels[b]

        if n_queries * n_cls > 10000:
            pre_topk = score.topk(10000).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

        keep_inds = batched_nms(box, score, lbls, 0.7)[:100]
        results.append(
            {
                "scores": score[keep_inds],
                "labels": lbls[keep_inds],
                "boxes": box[keep_inds],
            }
        )

    return results


def prepare_bbox_for_cropping(bbox, hw, padding=5, is_normalized=False):
    # prepares xyxy bbox
    new_boxes = []
    h, w = hw
    for i, boxes_padded in enumerate(bbox):
        boxes_padded = boxes_padded.clone()
        if is_normalized:
            assert hw is not None and len(hw) == 2
            boxes_padded[..., [0, 2]] *= hw[1]
            boxes_padded[..., [1, 3]] *= hw[0]
        else:
            if boxes_padded.max() < 1:
                print(f"WARNING: boxes seem normalized {boxes_padded=}")
        boxes_padded[..., 0] = boxes_padded[..., 0] - padding
        boxes_padded[..., 1] = boxes_padded[..., 1] - padding
        boxes_padded[..., 2] = boxes_padded[..., 2] + padding
        boxes_padded[..., 3] = boxes_padded[..., 3] + padding
        boxes_padded[..., 0].clamp_(min=0, max=w)
        boxes_padded[..., 1].clamp_(min=0, max=h)
        boxes_padded[..., 2].clamp_(min=0, max=w)
        boxes_padded[..., 3].clamp_(min=0, max=h)
        new_boxes.append(boxes_padded)
    return new_boxes


def get_crops(
    rgb,
    bbox_xyxy,
    hw,
    crop_size=(60, 80),
    padding=5,
    is_normalized=True,
):
    new_boxes = prepare_bbox_for_cropping(bbox_xyxy, hw=hw, is_normalized=is_normalized, padding=padding)
    rgb_crop = roi_align(
        rgb,
        new_boxes,
        crop_size,
    )
    return rgb_crop
