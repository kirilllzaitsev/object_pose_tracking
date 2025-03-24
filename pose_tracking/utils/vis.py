"""some code is from:
https://github.com/nv-nguyen/gigapose/blob/main/src/libVis/numpy.py#L137
https://github.com/NVlabs/FoundationPose/blob/main/Utils.py#L723
"""

import copy
import os

import cv2
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import torchvision
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw
from pose_tracking.dataset.ds_common import convert_seq_batch_to_batch_seq
from pose_tracking.metrics import world_to_cam
from pose_tracking.utils.common import (
    adjust_depth_for_plt,
    adjust_img_for_plt,
    cast_to_numpy,
    cast_to_torch,
)
from pose_tracking.utils.geom import (
    cam_to_2d,
    egocentric_delta_pose_to_pose,
    to_homo,
    world_to_2d,
    world_to_2d_pt_homo,
)
from pose_tracking.utils.kpt_utils import is_torch
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from pose_tracking.utils.video_utils import show_video
from skimage.feature import canny
from skimage.morphology import binary_dilation
from torchvision.ops.boxes import clip_boxes_to_image
from tqdm import tqdm


def draw_xyz_axis(rgb, rt, K, scale=10.0, thickness=2, transparency=0, is_input_rgb=False, do_add_text=False):
    """
    @color: BGR
    """
    if is_input_rgb:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float).reshape(-1, 1)
    yy = np.array([0, 1, 0, 1]).astype(float).reshape(-1, 1)
    zz = np.array([0, 0, 1, 1]).astype(float).reshape(-1, 1)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(world_to_2d_pt_homo(np.array([0.0, 0.0, 0.0, 1]).reshape(-1, 1), K, rt))
    xx = tuple(world_to_2d_pt_homo(xx, K, rt))
    yy = tuple(world_to_2d_pt_homo(yy, K, rt))
    zz = tuple(world_to_2d_pt_homo(zz, K, rt))
    line_type = cv2.LINE_AA

    color_x = (0, 0, 255)
    color_y = (255, 255, 0)
    color_z = (255, 0, 0)

    arrow_len = 0
    tmp = rgb.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, xx, color=color_x, thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, yy, color=color_y, thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, zz, color=color_z, thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp = tmp.astype(np.uint8)
    if is_input_rgb:
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    if do_add_text:
        origin_3d_cam = [round(x, 2) for x in world_to_cam(np.array([0.0, 0.0, 0.0]).reshape(-1, 1), rt)[0]]
        cv2.putText(
            tmp,
            f"{origin_3d_cam}",
            (origin[0], origin[1] - 80),
            color=0,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
        )
        rot_cam = rt[:3, :3]
        for i in range(3):
            cv2.putText(
                tmp,
                f"{[round(x, 2) for x in rot_cam[i]]}",
                (origin[0], origin[1] + 80 + i * 20),
                color=0,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
            )

    return tmp


def draw_posed_3d_box(img, rt, K, bbox, line_color=(0, 255, 0), linewidth=2):
    """Revised from 6pack dataset/inference_dataset_nocs.py::projection
    @bbox: (2,3) min/max
    @line_color: RGB
    """
    bbox = cast_to_numpy(bbox)
    min_xyz = bbox.min(axis=0)
    xmin, ymin, zmin = min_xyz
    max_xyz = bbox.max(axis=0)
    xmax, ymax, zmax = max_xyz

    def draw_line3d(start, end, img):
        pts = np.stack((start, end), axis=0).reshape(-1, 3)
        pts = (rt @ to_homo(pts).T).T[:, :3]  # (2,3)
        projected = (K @ pts.T).T
        uv = np.round(projected[:, :2] / projected[:, 2].reshape(-1, 1)).astype(int)  # (2,2)
        img = cv2.line(img, uv[0].tolist(), uv[1].tolist(), color=line_color, thickness=linewidth, lineType=cv2.LINE_AA)
        return img

    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start + np.array([xmax - xmin, 0, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start + np.array([0, ymax - ymin, 0])
            img = draw_line3d(start, end, img)

    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start + np.array([0, 0, zmax - zmin])
            img = draw_line3d(start, end, img)

    return img


def draw_poses_on_video(
    rgbs, intrinsics, poses_pred, poses_gt=None, bbox=None, bbox_color=(255, 255, 0), scale=0.05, take_n=None
):
    """
    Given a list of rgb images, camera intrinsics, CAD bounding box, and poses, draw the poses and object axes on the images. The args have to be numpy arrays.

    Args:
        rgbs: rgb images
        intrinsics: 3x3 camera intrinsics matrix
        poses_obj: 4x4 poses of the object
        bbox: 8x3 bounding box vertices
        bbox_color: rgb color
        scale: scale of the object axes (50 if in mm, 0.05 if in m)
    """
    images = []
    num_frames = min(len(rgbs), len(poses_pred)) if take_n is None else take_n
    for frame_idx in tqdm(range(num_frames), leave=True, desc="Frame"):
        rgb = rgbs[frame_idx]
        K = intrinsics[frame_idx] if isinstance(intrinsics, list) else intrinsics
        pose_pred = poses_pred[frame_idx]
        if poses_gt is not None:
            pose_gt = poses_gt[frame_idx]
        else:
            pose_gt = None
        rgb_with_pose = draw_pose_on_img(
            rgb, K, pose_pred, bbox=bbox, bbox_color=bbox_color, scale=scale, pose_gt=pose_gt
        )
        images.append(rgb_with_pose)
    images = np.array(images)
    return images


def draw_pose_on_img(
    rgb, K, pose_pred, bbox=None, bbox_color=(255, 255, 0), scale=50.0, pose_gt=None, final_frame=None
):
    if len(pose_pred.shape) == 3:
        final_frame = None
        if bbox is not None:
            assert len(bbox.shape) == 3, f"{bbox.shape=}"
        for idx in range(len(pose_pred)):
            final_frame = draw_pose_on_img(
                rgb,
                K,
                pose_pred[idx],
                bbox=None if bbox is None else bbox[idx],
                bbox_color=bbox_color,
                scale=scale,
                pose_gt=None if pose_gt is None else pose_gt[idx],
                final_frame=final_frame,
            )
        return final_frame

    rgb = adjust_img_for_plt(rgb) if final_frame is None else final_frame
    K = cast_to_numpy(K)
    pose_pred = cast_to_numpy(pose_pred)
    final_frame = draw_xyz_axis(rgb, scale=scale, K=K, rt=pose_pred, is_input_rgb=True)
    if bbox is not None:
        final_frame = draw_posed_3d_box(final_frame, rt=pose_pred, K=K, bbox=bbox, line_color=bbox_color)
        if pose_gt is not None:
            pose_gt = cast_to_numpy(pose_gt)
            final_frame = draw_posed_3d_box(final_frame, rt=pose_gt, K=K, bbox=bbox, line_color=(0, 255, 0))
    return final_frame


def vis_bbox_2d(
    img,
    bbox,
    color=(255, 0, 0),
    width=3,
    format="xyxy",
    is_normalized=False,
    label=None,
    score=None,
    label_place="top",
    final_frame=None,
):
    img = adjust_img_for_plt(img) if final_frame is None else final_frame
    bbox = cast_to_numpy(bbox).squeeze()

    if len(bbox.shape) == 2 and bbox.shape[-1] == 4:
        final_frame = None
        for idx in range(len(bbox)):
            final_frame = vis_bbox_2d(
                img,
                bbox[idx],
                color=color,
                width=width,
                format=format,
                is_normalized=is_normalized,
                label=label[idx] if label is not None else None,
                score=score[idx] if score is not None else None,
                label_place=label_place,
                final_frame=final_frame,
            )
        return final_frame

    img = np.ascontiguousarray(img)
    if bbox.shape == (4, 2):
        bbox_xy_ul = bbox[0]
        bbox_xy_br = bbox[2]
    elif bbox.shape == (2, 2):
        bbox_xy_ul = bbox[0]
        bbox_xy_br = bbox[1]
    elif format == "cxcywh":
        bbox_xyxy = box_cxcywh_to_xyxy(cast_to_torch(bbox))
        bbox_xyxy = cast_to_numpy(bbox_xyxy)
        bbox_xy_ul = bbox_xyxy[:2]
        bbox_xy_br = bbox_xyxy[2:]
    else:
        bbox_xy_ul = bbox[:2]
        bbox_xy_br = bbox[2:]

    if is_normalized:
        h, w = img.shape[:2]
        bbox_xy_ul = bbox_xy_ul * np.array([w, h])
        bbox_xy_br = bbox_xy_br * np.array([w, h])
    img = cv2.rectangle(
        img,
        tuple(bbox_xy_ul.astype(int)),
        tuple(bbox_xy_br.astype(int)),
        color,
        width,
    )
    if label is not None:
        text = f"{label}"
        if score is not None:
            text += f", {score:.2f}"
        if label_place == "top":
            label_xy = (int(bbox_xy_ul[0]), int(bbox_xy_ul[1]) - 10)
        else:
            label_xy = (int(bbox_xy_br[0]), int(bbox_xy_br[1]) + 10)
        img = cv2.putText(
            img,
            text,
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return img


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def plot_kpt_matches(img0, img1, mkpts0, mkpts1, color=None, kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    mkpts0 = cast_to_numpy(mkpts0)
    mkpts1 = cast_to_numpy(mkpts1)
    if color is None:
        color = np.random.rand(len(mkpts0), 3)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                transform=fig.transFigure,
                c=color[i],
                linewidth=1,
            )
            for i in range(len(mkpts0))
        ]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    text = text or [f"{len(mkpts0)} matches"]
    fig.text(
        0.01, 0.99, "\n".join(text), transform=fig.axes[0].transAxes, fontsize=15, va="top", ha="left", color=txt_color
    )

    # save or return figure
    if path:
        fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
    return fig


def vis_kpts(img_PIL, points_2d, color=(0, 255, 0), do_fix_img_color=False, conf=None, include_ids=False):
    if conf is not None:
        sorted, indices = torch.sort(conf)

    img = adjust_img_for_plt(img_PIL)
    if do_fix_img_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points_2d = cast_to_numpy(points_2d).astype(int)
    for idx, point in enumerate(points_2d):
        size = 3 if conf is None else int(3 + 3 * conf[idx] * 2)
        img = cv2.circle(img, tuple(point), size, color, -1)

    img = cv2.putText(
        img,
        f"{len(points_2d)} kpts",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if include_ids:
        for idx, point in enumerate(points_2d):
            img = cv2.putText(
                img,
                str(idx),
                tuple(point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return np.array(img)


def add_border(image, color=[255, 0, 0], border_size=5):
    image[:border_size, :] = color
    image[-border_size:, :] = color

    # Add the border to the left and right columns
    image[:, :border_size] = color
    image[:, -border_size:] = color
    return image


def write_text_on_image(image, text, color=[255, 0, 0]):
    image_size = image.shape[:2]
    # write text on top left corner
    position = (image_size[1] // 15, image_size[0] // 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_contour(img, mask, color, dilation_footprint_shape=(1, 1)):
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones(dilation_footprint_shape))
    img[edge, :] = color
    return img


def overlay_mask_on_rgb_contour(rgb, mask, gray=False, color=(255, 255, 0), alpha=0.5):
    if gray:
        gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        img = np.array(rgb)
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])
    img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
    img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
    img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]
    img = draw_contour(img, mask, color=color)
    return img


def overlay_mask_on_rgb(rgb, mask, alpha=0.5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb)
    ax.imshow(mask, alpha=alpha)
    return ax


def mask_background(gray_img, color_img, masks, color=(255, 0, 0), contour=True):
    """
    Put the color in the gray image according to the mask and add the contour
    """
    if isinstance(gray_img, Image.Image):
        gray_img = np.array(gray_img)
    for mask in masks:
        gray_img[mask > 0, :] = color_img[mask > 0, :]
        if contour:
            gray_img = draw_contour(gray_img, mask, color=color, to_pil=False)
    return gray_img


def PIL_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def make_grid_image(imgs, nrow=None, padding=5, pad_value=255, dtype=np.uint8, use_existing_fig=False):
    """
    @imgs: (B,H,W,C) np array
    @nrow: num of images per row
    """
    if len(imgs) == 0:
        print("No images to plot")
    imgs = torch.as_tensor(np.asarray(imgs))
    if imgs.shape[-1] == 3:
        imgs = imgs.permute(0, 3, 1, 2)
    if nrow is None:
        n = imgs.size(0)
        for i in range(6, 1, -1):
            if n % i == 0:
                nrow = i
                break
        else:
            nrow = 1
    nrow = min(len(imgs), max(nrow, 1))
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=padding, pad_value=pad_value)
    grid = grid.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(dtype)
    if not use_existing_fig:
        plt.figure(figsize=(nrow * 5, (len(imgs) // nrow) * 5))
        plt.axis("off")
        plt.imshow(grid)
        plt.show()
    return grid


def plot_seq(
    seq,
    keys_to_plot=None,
    take_n=None,
    batch_idx=0,
    bbox_format="xyxy",
    bbox_is_normalized=False,
    use_label=False,
    rot_repr="quaternion",
    nrow=None,
):
    keys_to_plot = keys_to_plot or []
    target_key = "target" if "target" in seq[0] and len(seq[0]["target"]) else "targets"

    def fetcher_fn(k, sidx=0):
        if not key_in_seq(k):
            raise ValueError(f"{k} not found in the sequence")
        if key_in_target(k):
            val = seq[sidx][target_key]
            if isinstance(val, dict):
                return val[k][batch_idx]
            else:
                return val[batch_idx][k]
        else:
            return seq[sidx][k][batch_idx]

    def key_in_seq(k):
        return seq[0].get(k, []) != [] or key_in_target(k)

    def key_in_target(k):
        if target_key in seq[0]:
            val = seq[0][target_key]
            if isinstance(val, dict):
                return len(val.get(k, [])) > 0
            else:
                return val[batch_idx].get(k, []) != []
        return False

    if isinstance(seq, list) and isinstance(seq[0], list):
        seq = seq[batch_idx]
        print(f"taking {batch_idx=} of {len(seq)}")
    if any([x in seq for x in ["image", "rgb"]]):
        batch_seq = convert_seq_batch_to_batch_seq(
            seq, keys=keys_to_plot + ["rgb", "image", "intrinsics", "mesh_bbox", target_key]
        )
        seq = batch_seq
        seq = [x[batch_idx] for x in batch_seq]
        print(f"taking {batch_idx=} of {len(batch_seq)}")
    img_key = "rgb" if "rgb" in seq[0] and len(seq[0]["rgb"]) > 0 else "image"
    first_key = img_key
    if len(keys_to_plot) == 0:
        keys_to_plot.append(img_key)

    if len(seq) > 20:
        print(f"Taking first 20 frames of {len(seq)=}")
        seq = seq[:20]
    print(f"{len(seq)=}")
    take_n = min(take_n, len(seq)) if take_n is not None else len(seq)
    results = {}
    if any("pose" in key for key in keys_to_plot):
        if first_key in keys_to_plot:
            keys_to_plot = [key for key in keys_to_plot if key != first_key]
    if len(seq[0][img_key].shape) == 4:
        print(f"Taking {batch_idx} image from the batch of size {seq[0][img_key].shape[0]}")
    else:
        batch_idx = slice(None)

    for key in keys_to_plot:
        arr = []
        for sidx in range(take_n):
            img = fetcher_fn(key, sidx)
            dtype = np.uint8
            if key in ["depth"]:
                grid_img = img
                dtype = np.float32
            elif key in ["mask"]:
                grid_img = adjust_img_for_plt(img[None])
            elif "bbox_3d" in key:
                pts_3d = fetcher_fn(key, sidx)
                pts_2d = cam_to_2d(pts_3d, fetcher_fn("intrinsics", sidx).float())
                grid_img = vis_kpts(fetcher_fn(img_key, sidx), pts_2d, color=(0, 255, 0), do_fix_img_color=True)

            elif "box" in key:
                label = None
                for lkey in ["labels", "class_id"]:
                    if use_label and key_in_seq(lkey):
                        label = fetcher_fn(lkey, sidx)
                grid_img = vis_bbox_2d(
                    fetcher_fn(img_key, sidx),
                    img,
                    format=bbox_format,
                    is_normalized=bbox_is_normalized,
                    label=label,
                )
            elif "pose" in key:
                pose = img
                if pose.shape[-2:] != (4, 4):
                    pose = convert_pose_vector_to_matrix(pose, rot_repr=rot_repr)

                if key in ["pose_mat_prev_gt_abs"]:
                    t_gt_rel, rot_gt_rel_mat = fetcher_fn("t_gt_rel", sidx), fetcher_fn("rot_gt_rel_mat", sidx)
                    pose = egocentric_delta_pose_to_pose(pose, t_gt_rel, rot_gt_rel_mat)

                grid_img = draw_pose_on_img(
                    fetcher_fn(img_key, sidx),
                    fetcher_fn("intrinsics", sidx),
                    pose,
                    bbox=fetcher_fn("mesh_bbox", sidx),
                    bbox_color=(255, 255, 0),
                    scale=0.15,
                )
                dtype = np.uint8
            else:
                grid_img = adjust_img_for_plt(img)
                dtype = np.uint8
            arr.append(grid_img)
        res = make_grid_image(arr, nrow=nrow, padding=5, dtype=dtype, use_existing_fig=True)
        plt.figure(figsize=(5 * 5, 5 * take_n))
        plt.imshow(res)
        plt.axis("off")
        results[key] = res


def get_cmap(np_img):
    cmap = matplotlib.colormaps.get_cmap("magma")
    tmp = cmap(np_img)[..., :3]
    return tmp


def plot(fn, num_cols=1):

    def inner(*args, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, num_cols, figsize=(10, 5))
            ax.axis("off")
        res = fn(*args, **kwargs)
        ax.imshow(res)
        return ax

    return inner


@plot
def plot_bbox_2d(img, bbox, format="xyxy", is_normalized=False, **kwargs):
    return vis_bbox_2d(img, bbox, format=format, is_normalized=is_normalized, **kwargs)


@plot
def plot_kpts(img_PIL, points_2d, color=(0, 255, 0), **kwargs):
    return vis_kpts(img_PIL, points_2d, color, **kwargs)


@plot
def plot_rgb(color):
    return adjust_img_for_plt(color)


@plot
def plot_optical_flow(flow):
    return vis_optical_flow(flow)


@plot
def plot_normals(flow):
    return vis_normals(flow)


@plot
def plot_pose(color, pose, K, bbox=None, scale=0.05, bbox_color=(255, 255, 0)):
    return vis_pose(color, pose, K, bbox=bbox, scale=scale, bbox_color=bbox_color)


def plot_rgb_depth(color, depth, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_rgb(color, ax=axs[0])
    plot_depth(depth, ax=axs[1])
    return axs


def plot_depth(depth, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.axis("off")
    depth = adjust_depth_for_plt(depth)
    im = ax.imshow(depth, cmap="viridis")
    plt.colorbar(im, ax=ax)
    return ax


def vis_normals(x):
    rgb = (x + 1) / 2
    return adjust_img_for_plt(rgb)


def vis_optical_flow(flow):
    h, w = flow.shape[:2]
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2  # Angle of flow
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude of flow
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_tracks(video, pred_tracks, pred_visibility, name="queries", save_dir="/tmp/videos"):
    from cotracker.utils.visualizer import Visualizer, read_video_from_path

    vis = Visualizer(save_dir=save_dir, linewidth=6, mode="cool", tracks_leave_trace=-1)
    vis.visualize(video=video[None], tracks=pred_tracks, visibility=pred_visibility, filename=name)
    return show_video(f"{save_dir}/{name}.mp4")


def plot_imgs(imgs, n_samples=15, return_fig=False):
    ncols = min(5, len(imgs))
    n_samples = min(n_samples, len(imgs))
    nrows = n_samples // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    for i in range(nrows):
        for j in range(ncols):
            if nrows > 1:
                ax = axs[i, j]
            elif ncols > 1:
                ax = axs[j]
            else:
                ax = axs
            # overlay_mask_on_rgb(rgb, mask, ax=ax)
            input_ = adjust_img_for_plt(imgs[i * ncols + j])
            ax.imshow(adjust_img_for_plt(input_))
            ax.axis("off")
            ax.set_title(f"Frame {i*ncols+j+1}")
    plt.tight_layout()
    if return_fig:
        return fig


def plot_sample_dict(sample):
    rgb = sample["rgb"].squeeze()
    depth = sample["depth"].squeeze()
    mask = sample["mask"].squeeze()
    pose = sample["pose"].squeeze()
    if pose.shape[-1] == 7:
        pose = convert_pose_vector_to_matrix(pose)
    elif pose.shape[-1] == 9:
        pose = convert_pose_vector_to_matrix(pose, rot_repr="rotation6d")
    if "bbox_2d" in sample:
        bbox_2d = sample["bbox_2d"].squeeze()
    else:
        bbox_2d = None
    return plot_sample(
        rgb, depth, mask, pose=pose, K=sample.get("intrinsics"), bbox=sample.get("mesh_bbox"), bbox_2d=bbox_2d
    )


def plot_sample(rgb, depth, mask, pose=None, K=None, bbox=None, bbox_2d=None, scale=0.05):
    ncols = 2
    use_pose = pose is not None
    if use_pose:
        ncols = 3
    fig, axs = plt.subplots(2, ncols, figsize=(5 * ncols, 5 * ncols))
    color = adjust_img_for_plt(rgb)
    depth = adjust_depth_for_plt(depth)
    mask = adjust_img_for_plt(mask)
    axs[0, 0].imshow(color)
    im = axs[0, 1].imshow(depth, cmap="viridis")
    fig.colorbar(im, ax=axs[0, 1])
    axs[1, 0].imshow(mask)
    color_masked = copy.deepcopy(color)
    color_masked[mask == 0] = 0
    axs[1, 1].imshow(color_masked)
    if use_pose:
        assert K is not None
        img_pose = vis_pose(color, pose, K, bbox=bbox, scale=scale)
        axs[0, 2].imshow(img_pose)
    if bbox_2d is not None:
        img_bbox_2d = vis_bbox_2d(color, bbox_2d)
        axs[1, 2].imshow(img_bbox_2d)
    return fig, axs


def plot_sample_pose_dict(sample, scale=0.05, bbox=None, ax=None):
    color = sample["rgb"]
    pose = sample["pose"]
    bbox = sample.get("mesh_bbox") if bbox is None else bbox
    if pose.shape[-1] == 7:
        pose = convert_pose_vector_to_matrix(pose)
    K = sample["intrinsics"]
    return plot_pose(color, pose, K, bbox=bbox, ax=ax, scale=scale)


def vis_pose_pred(pose_pred, pose_gt=None, *args, **kwargs):
    use_gt = pose_gt is not None
    if use_gt:
        kwargs["bbox_color"] = (255, 255, 0)
    img = vis_pose(*args, pose=pose_pred, **kwargs)
    if use_gt:
        kwargs["bbox_color"] = (0, 255, 0)
        kwargs["color"] = img
        img = vis_pose(pose=pose_gt, **kwargs)
    return img


def vis_pose(color, pose, K, bbox=None, scale=0.05, bbox_color=(255, 255, 0)):
    color = adjust_img_for_plt(color)
    color_with_pose = draw_pose_on_img(color, K, pose, bbox=bbox, bbox_color=bbox_color, scale=scale)
    return color_with_pose


def render_offscreen(mesh, obj_pose, intrinsic, w, h, headless=False):
    # https://github.com/nv-nguyen/bop_viz_kit
    if headless:
        os.environ["DISPLAY"] = ":1"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    cam_pose = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 0.0]),
        ambient_light=np.array([0.2, 0.2, 0.2, 1.0]),
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=4.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000)
    scene.add(light, pose=cam_pose)
    # set camera pose from openGL to openCV pose
    scene.add(camera, pose=cam_pose)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh, pose=obj_pose)
    r = pyrender.OffscreenRenderer(w, h)
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    # flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.RGBA
    flags = pyrender.RenderFlags.OFFSCREEN
    color, depth = r.render(scene, flags=flags)
    # color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)  # RGBA to BGRA (for OpenCV)
    return color, depth


def draw_pose_contour(cvImg, mesh, intrinsic, obj_openCV_pose, color, thickness=3, headless=False):
    rendered_color, depth = render_offscreen(mesh, obj_openCV_pose, intrinsic, w=640, h=480, headless=headless)
    validMap = (depth > 0).astype(np.uint8)
    contours, _ = cv2.findContours(validMap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cvImg = cv2.drawContours(cvImg, contours, -1, color, thickness)
    return rendered_color, cvImg


def plot_bbox_2d_plt(bbox_2d, color="r"):
    plt.plot(bbox_2d[[0, 1], 0], bbox_2d[[0, 1], 1], color)
    plt.plot(bbox_2d[[1, 2], 0], bbox_2d[[1, 2], 1], color)
    plt.plot(bbox_2d[[2, 3], 0], bbox_2d[[2, 3], 1], color)
    plt.plot(bbox_2d[[3, 0], 0], bbox_2d[[3, 0], 1], color)


def vis_res_tracking_text(rgb, result, target, tracking):

    legends = []

    num_track_queries = num_track_queries_with_id = 0
    if tracking:
        num_track_queries = len(target["track_query_boxes"])
        num_track_queries_with_id = len(target["track_query_match_ids"])
        track_ids = target["track_ids"][target["track_query_match_ids"]]

    keep = result["scores"].cpu() > result["scores_no_object"].cpu()
    # keep = torch.ones_like(result['scores'].cpu())

    cmap = plt.cm.get_cmap("hsv", len(keep))

    prop_i = 0
    for box_id in range(len(keep)):
        rect_color = "green"
        offset = 0
        text = f"{result['scores'][box_id]:0.2f}"

        if tracking:
            if target["track_queries_fal_pos_mask"][box_id]:
                rect_color = "red"
            elif target["track_queries_mask"][box_id]:
                offset = 50
                rect_color = "blue"
                text = (
                    f"- track_id {track_ids[prop_i]}\n"
                    f"- cls_score {text}\n"
                    f"- iou {result['track_queries_with_id_iou'][prop_i]:0.2f}"
                )
                prop_i += 1

        if not keep[box_id]:
            continue

        legends.append(text)

    legends = "\n".join(legends)

    stats_text = ""

    query_keep = keep
    if tracking:
        query_keep = keep[target["track_queries_mask"] == 0]

    stats_text += f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n"

    if num_track_queries:
        track_queries_label = (
            f"- track queries ({keep[target['track_queries_mask']].sum() - keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries_with_id})"
        )

        stats_text += track_queries_label

    if num_track_queries_with_id != num_track_queries:
        track_queries_fal_pos_label = (
            f"- false track queries ({keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries - num_track_queries_with_id})"
        )

        stats_text += track_queries_fal_pos_label

    tracks_prev_frame = []
    for frame_prefix in ["prev", "prev_prev"]:
        # if f'{frame_prefix}_image_id' not in target or f'{frame_prefix}_boxes' not in target:
        if f"{frame_prefix}_target" not in target:
            continue

        frame_target = target[f"{frame_prefix}_target"]
        cmap = plt.cm.get_cmap("hsv", len(frame_target["track_ids"]))

        tracks_prev_frame_ = []
        for j, track_id in enumerate(frame_target["track_ids"]):
            tracks_prev_frame_.append(
                {
                    "boxes": frame_target["boxes"][j],
                    "track_id": track_id,
                }
            )

        tracks_prev_frame.append(tracks_prev_frame_)

    # draw all data on respective imgs

    figure, axarr = plt.subplots(len(rgb), figsize=(20, 20))
    figure.tight_layout()

    for idx, (ax, img) in enumerate(zip(axarr, rgb)):
        ax.set_axis_off()
        ax.imshow(adjust_img_for_plt(img))

        # add legend
        ax.text(0, 0, legends[idx], fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

    # add stats
    axarr[0].text(0, 0, stats_text, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.axis("off")

    img = fig_to_numpy(figure).transpose(2, 0, 1)
    plt.close()

    return {"legends": legends, "stats_text": stats_text, "tracks_prev_frame": tracks_prev_frame, "img": img}


def vis_res_tracking(rgb, result, target, tracking):
    imgs = [adjust_img_for_plt(rgb)]
    img_ids = [target["image_id"].item()]
    for key in ["prev", "prev_prev"]:
        if f"{key}_image" in target:
            imgs.append(adjust_img_for_plt(target[f"{key}_image"]))
            img_ids.append(target[f"{key}_target"][f"image_id"].item())

    # img.shape=[3, H, W]
    dpi = 96
    figure, axarr = plt.subplots(len(imgs), figsize=(10, 10))
    figure.tight_layout()
    figure.set_dpi(dpi)
    # figure.set_size_inches(imgs[0].shape[2] / dpi, imgs[0].shape[1] * len(imgs) / dpi)

    if len(imgs) == 1:
        axarr = [axarr]

    for ax, img, img_id in zip(axarr, imgs, img_ids):
        ax.set_axis_off()
        ax.imshow(img)

        ax.text(0, 0, f"IMG_ID={img_id}", fontsize=20, bbox=dict(facecolor="white", alpha=0.5))

    num_track_queries = num_track_queries_with_id = 0
    if tracking:
        num_track_queries = len(target["track_query_boxes"])
        num_track_queries_with_id = len(target["track_query_match_ids"])
        track_ids = target["track_ids"][target["track_query_match_ids"]]

    keep = result["scores"].cpu() > result["scores_no_object"].cpu()
    # keep = torch.ones_like(result['scores'].cpu())

    cmap = plt.cm.get_cmap("hsv", len(keep))

    prop_i = 0
    for box_id in range(len(keep)):
        rect_color = "green"
        offset = 0
        text = f"{result['scores'][box_id]:0.2f}"

        if tracking:
            if target["track_queries_fal_pos_mask"][box_id]:
                rect_color = "red"
            elif target["track_queries_mask"][box_id]:
                offset = 50
                rect_color = "blue"
                text = f"{track_ids[prop_i]}\n" f"{text}\n" f"{result['track_queries_with_id_iou'][prop_i]:0.2f}"
                prop_i += 1

        if not keep[box_id]:
            continue

        # x1, y1, x2, y2 = result['boxes'][box_id]
        result_boxes = clip_boxes_to_image(result["boxes"], target["size"])
        x1, y1, x2, y2 = result_boxes[box_id]

        axarr[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=rect_color, linewidth=2))

        axarr[0].text(x1, y1 + offset, text, fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

        if "masks" in result:
            mask = result["masks"][box_id][0].numpy()
            mask = np.ma.masked_where(mask == 0.0, mask)

            axarr[0].imshow(mask, alpha=0.5, cmap=colors.ListedColormap([cmap(box_id)]))

    query_keep = keep
    if tracking:
        query_keep = keep[target["track_queries_mask"] == 0]

    legend_handles = [
        mpatches.Patch(
            color="green",
            label=f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n- cls_score",
        )
    ]

    if num_track_queries:
        track_queries_label = (
            f"track queries ({keep[target['track_queries_mask']].sum() - keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries_with_id})\n- track_id\n- cls_score\n- iou"
        )

        legend_handles.append(mpatches.Patch(color="blue", label=track_queries_label))

    if num_track_queries_with_id != num_track_queries:
        track_queries_fal_pos_label = (
            f"false track queries ({keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries - num_track_queries_with_id})"
        )

        legend_handles.append(mpatches.Patch(color="red", label=track_queries_fal_pos_label))

    axarr[0].legend(handles=legend_handles)

    i = 1
    for frame_prefix in ["prev", "prev_prev"]:
        # if f'{frame_prefix}_image_id' not in target or f'{frame_prefix}_boxes' not in target:
        if f"{frame_prefix}_target" not in target:
            continue

        frame_target = target[f"{frame_prefix}_target"]
        cmap = plt.cm.get_cmap("hsv", len(frame_target["track_ids"]))

        for j, track_id in enumerate(frame_target["track_ids"]):
            x1, y1, x2, y2 = frame_target["boxes"][j]
            axarr[i].text(x1, y1, f"track_id={track_id}", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))
            axarr[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="green", linewidth=2))

            if "masks" in frame_target:
                mask = frame_target["masks"][j].cpu().numpy()
                mask = np.ma.masked_where(mask == 0.0, mask)

                axarr[i].imshow(mask, alpha=0.5, cmap=colors.ListedColormap([cmap(j)]))
        i += 1

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.axis("off")

    img = fig_to_numpy(figure).transpose(2, 0, 1)
    plt.close()

    return img


def fig_to_numpy(fig):
    w, h = fig.get_size_inches() * fig.dpi
    w = int(w.item())
    h = int(h.item())
    canvas = FigureCanvas(fig)
    canvas.draw()
    numpy_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(h, w, 3)
    return np.copy(numpy_image)
