import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pose_tracking.utils.common import cast_to_numpy


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_2d_bbox(bbox, ax):
    x0, y0 = bbox[0], bbox[1]
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_2d_bbox(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def convert_mask_to_xyxy_box(mask, offset=20):
    h, w = mask.shape
    y, x = np.where(mask)
    x1, y1 = x.min(), y.min()
    x2, y2 = x.max(), y.max()
    x1 = max(0, x1 - offset)
    y1 = max(0, y1 - offset)
    x2 = min(w, x2 + offset)
    y2 = min(h, y2 + offset)
    crop_bbox = [int(x1), int(y1), int(x2), int(y2)]
    return crop_bbox


def mask_erode(prev_mask, kernel_size=11):
    is_tensor = isinstance(prev_mask, torch.Tensor)
    device = prev_mask.device if is_tensor else None
    prev_mask = cast_to_numpy(prev_mask, dtype=np.uint8)
    res = cv2.erode(prev_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    if is_tensor:
        res = torch.from_numpy(res).to(device)
    return res
