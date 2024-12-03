"""some code is from:
https://github.com/nv-nguyen/gigapose/blob/main/src/libVis/numpy.py#L137
https://github.com/NVlabs/FoundationPose/blob/main/Utils.py#L723
"""

import copy
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import torchvision
from PIL import Image, ImageDraw
from pose_tracking.utils.common import (
    adjust_depth_for_plt,
    adjust_img_for_plt,
    cast_to_numpy,
)
from pose_tracking.utils.geom import to_homo, world_to_2d_pt_homo
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.video_utils import show_video
from skimage.feature import canny
from skimage.morphology import binary_dilation
from tqdm.auto import tqdm


def draw_xyz_axis(rgb, rt, K, scale=10.0, thickness=2, transparency=0, is_input_rgb=False):
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


def draw_pose_on_img(rgb, K, pose_pred, bbox=None, bbox_color=(255, 255, 0), scale=50.0, pose_gt=None):
    rgb = adjust_img_for_plt(rgb)
    K = cast_to_numpy(K)
    pose_pred = cast_to_numpy(pose_pred)
    final_frame = draw_xyz_axis(rgb, scale=scale, K=K, rt=pose_pred, is_input_rgb=True)
    if bbox is not None:
        final_frame = draw_posed_3d_box(final_frame, rt=pose_pred, K=K, bbox=bbox, line_color=bbox_color)
        if pose_gt is not None:
            pose_gt = cast_to_numpy(pose_gt)
            final_frame = draw_posed_3d_box(final_frame, rt=pose_gt, K=K, bbox=bbox, line_color=(0, 255, 0))
    return final_frame


def plot_bbox_2d(img, bbox, ax=None, format="xyxy", is_normalized=False, **kwargs):
    img = vis_bbox_2d(img, bbox, format=format, is_normalized=is_normalized, **kwargs)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(img)
    return ax


def vis_bbox_2d(img, bbox, color=(255, 0, 0), width=3, format="xyxy", is_normalized=False):
    img = np.ascontiguousarray(adjust_img_for_plt(img))
    if bbox.shape == (4, 2):
        bbox_xy_bl = bbox[0]
        bbox_xy_ur = bbox[2]
    elif bbox.shape == (2, 2):
        bbox_xy_bl = bbox[0]
        bbox_xy_ur = bbox[1]
    else:
        bbox_xy_bl = bbox[:2]
        bbox_xy_ur = bbox[2:]
    if format == "cxcywh":
        bbox_xyxy = box_cxcywh_to_xyxy(bbox)
        bbox_xy_bl = bbox_xyxy[:2]
        bbox_xy_ur = bbox_xyxy[2:]

        if is_normalized:
            h, w = img.shape[:2]
            bbox_xy_bl = bbox_xy_bl * np.array([w, h])
            bbox_xy_ur = bbox_xy_ur * np.array([w, h])
    bbox_xy_ur = cast_to_numpy(bbox_xy_ur)
    bbox_xy_bl = cast_to_numpy(bbox_xy_bl)
    print(f"{bbox_xy_bl=}")
    print(f"{bbox_xy_ur=}")
    img = cv2.rectangle(
        img,
        tuple(bbox_xy_bl.astype(int)),
        tuple(bbox_xy_ur.astype(int)),
        color,
        width,
    )
    for pt in [bbox_xy_bl, bbox_xy_ur]:
        img = cv2.circle(img, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
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


def plot_kpts(img_PIL, points_2d, color="blue"):
    img_PIL = Image.fromarray(adjust_img_for_plt(img_PIL))
    points_2d = cast_to_numpy(points_2d)
    draw = ImageDraw.Draw(img_PIL)
    for point in points_2d:
        draw.rectangle(
            ((point[0] - 0.1, point[1] + 0.1), (point[0] - 0.1, point[1] + 0.1)),
            outline=color,
            width=5,
        )
    return img_PIL


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


def make_grid_image(imgs, nrow=5, padding=5, pad_value=255, dtype=np.uint8):
    """
    @imgs: (B,H,W,C) np array
    @nrow: num of images per row
    """
    grid = torchvision.utils.make_grid(
        torch.as_tensor(np.asarray(imgs)).permute(0, 3, 1, 2), nrow=nrow, padding=padding, pad_value=pad_value
    )
    grid = grid.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(dtype)
    return grid


def plot_seq(seq, keys_to_plot=["rgb"], take_n=None):
    take_n = take_n or len(seq)
    results = {}
    if len(seq[0]["rgb"].shape) == 4:
        subidx = 0
        print(f"Taking the first image from the batch of size {seq[0]['rgb'].shape[0]}")
    else:
        subidx = ...
    for key in keys_to_plot:
        arr = []
        for i in range(take_n):
            img = seq[i][key]
            if key in ["depth"]:
                grid_img = adjust_depth_for_plt(img[subidx])
                dtype = np.float32
            elif key in ["mask"]:
                grid_img = adjust_img_for_plt(img[subidx][None])
                dtype = np.uint8
            elif key in ["pose_mat_pred", "pose_mat_pred_abs", "pose_mat_gt_abs"]:
                grid_img = draw_pose_on_img(
                    seq[i]["rgb"][subidx],
                    seq[i]["intrinsics"][subidx],
                    img[subidx],
                    bbox=seq[i]["mesh_bbox"][subidx],
                    bbox_color=(255, 255, 0),
                    scale=0.1,
                )
                dtype = np.uint8
            else:
                grid_img = adjust_img_for_plt(img[subidx])
                dtype = np.uint8
            arr.append(grid_img)
        res = make_grid_image(arr, nrow=5, padding=5, dtype=dtype)
        plt.figure(figsize=(20, 20))
        plt.imshow(res)
        plt.axis("off")
        results[key] = res

    return results


def get_cmap(np_img):
    cmap = matplotlib.colormaps.get_cmap("magma")
    tmp = cmap(np_img)[..., :3]
    return tmp


def plot_rgb_depth(color, depth, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    color = adjust_img_for_plt(color)
    axs[0].imshow(color)
    plot_depth(depth, ax=axs[1])
    return axs


def plot_depth(depth, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    depth = adjust_depth_for_plt(depth)
    im = ax.imshow(depth, cmap="viridis")
    plt.colorbar(im, ax=ax)
    return ax


def vis_optical_flow(flow):
    h, w = flow.shape[:2]
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2  # Angle of flow
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude of flow
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_optical_flow(flow):
    rgb = vis_optical_flow(flow)
    plt.imshow(rgb)
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
    return plot_sample(rgb, depth, mask)


def plot_sample(rgb, depth, mask):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
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
    return fig, axs


def plot_sample_pose_dict(sample, scale=0.05, bbox=None, ax=None):
    color = sample["rgb"]
    pose = sample["pose"]
    if pose.shape[-1] == 7:
        pose = convert_pose_quaternion_to_matrix(pose)
    K = sample["intrinsics"]
    return plot_pose(color, pose, K, bbox=bbox, ax=ax, scale=scale)


def plot_pose(color, pose, K, bbox=None, ax=None, scale=0.05):
    color_with_pose = vis_pose(color, pose, K, bbox=bbox, scale=scale)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(color_with_pose)
    return ax


def vis_pose(color, pose, K, bbox=None, scale=0.05):
    color = adjust_img_for_plt(color)
    color_with_pose = draw_pose_on_img(color, K, pose, bbox=bbox, bbox_color=(255, 255, 0), scale=scale)
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
