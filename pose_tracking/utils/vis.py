"""some code is from:
https://github.com/nv-nguyen/gigapose/blob/main/src/libVis/numpy.py#L137
https://github.com/NVlabs/FoundationPose/blob/main/Utils.py#L723
"""

import copy

import cv2
import matplotlib
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from pose_tracking.utils.geom import project_3d_to_2d, to_homo
from skimage.feature import canny
from skimage.morphology import binary_dilation


def draw_xyz_axis(rgb, rt, K, scale=10, thickness=3, transparency=0, is_input_rgb=False):
    """
    @color: BGR
    """
    if is_input_rgb:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    xx = np.array([1, 0, 0, 1]).astype(float)
    yy = np.array([0, 1, 0, 1]).astype(float)
    zz = np.array([0, 0, 1, 1]).astype(float)
    xx[:3] = xx[:3] * scale
    yy[:3] = yy[:3] * scale
    zz[:3] = zz[:3] * scale
    origin = tuple(project_3d_to_2d(np.array([0, 0, 0, 1]), K, rt))
    xx = tuple(project_3d_to_2d(xx, K, rt))
    yy = tuple(project_3d_to_2d(yy, K, rt))
    zz = tuple(project_3d_to_2d(zz, K, rt))
    line_type = cv2.LINE_AA
    arrow_len = 0
    tmp = rgb.copy()
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, xx, color=(0, 0, 255), thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, yy, color=(0, 255, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len
    )
    mask = np.linalg.norm(tmp1 - tmp, axis=-1) > 0
    tmp[mask] = tmp[mask] * transparency + tmp1[mask] * (1 - transparency)
    tmp1 = tmp.copy()
    tmp1 = cv2.arrowedLine(
        tmp1, origin, zz, color=(255, 0, 0), thickness=thickness, line_type=line_type, tipLength=arrow_len
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


def draw_bbox(img, rt, K, bbox, line_color=(0, 255, 0), linewidth=2):
    def search_fit(points):
        """
        @points: (N,3)
        """
        min_x = min(points[:, 0])
        max_x = max(points[:, 0])
        min_y = min(points[:, 1])
        max_y = max(points[:, 1])
        min_z = min(points[:, 2])
        max_z = max(points[:, 2])
        return [min_x, max_x, min_y, max_y, min_z, max_z]

    def build_frame(min_x, max_x, min_y, max_y, min_z, max_z):
        bbox = []
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, min_y, max_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, min_z])
        for i in np.arange(min_x, max_x, 1.0):
            bbox.append([i, max_y, max_z])

        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([min_x, i, max_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, min_z])
        for i in np.arange(min_y, max_y, 1.0):
            bbox.append([max_x, i, max_z])

        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([min_x, max_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, min_y, i])
        for i in np.arange(min_z, max_z, 1.0):
            bbox.append([max_x, max_y, i])
        bbox = np.array(bbox)
        return bbox

    cam_cx = K[0, 2]
    cam_cy = K[1, 2]
    cam_fx = K[0, 0]
    cam_fy = K[1, 1]

    target_r = rt[:3, :3]
    target_t = rt[:3, 3]

    target = copy.deepcopy(bbox)
    limit = search_fit(target)
    bbox = build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5])

    bbox = np.dot(bbox, target_r.T) + target_t

    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for tg in bbox:
        y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        x = int(tg[1] * cam_fy / tg[2] + cam_cy)

        if x - linewidth < 0 or x + linewidth > 479 or y - linewidth < 0 or y + linewidth > 639:
            continue

        for xxx in range(x - linewidth + 1, x + linewidth):
            for yyy in range(y - linewidth + 1, y + linewidth):
                vis[xxx][yyy] = line_color

    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis


def draw_bbox_pil(img_PIL, bbox, color="red", width=3):
    draw = ImageDraw.Draw(img_PIL)
    draw.rectangle(
        (
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
        ),
        outline=color,
        width=width,
    )
    return img_PIL


def plot_kpt_matches(
    src_img,
    src_pts,
    tar_img,
    tar_pts,
    border_color,
    patch_size=14,
    concate_input=True,
    write_num_matches=True,
):
    if patch_size != 1:
        src_pts = np.array(src_pts) * patch_size  # + patch_size * 0.5
        tar_pts = np.array(tar_pts) * patch_size  # + patch_size * 0.5
    src_pts = [cv2.KeyPoint(x, y, 1) for x, y in np.float32(src_pts)]
    tar_pts = [cv2.KeyPoint(x, y, 1) for x, y in np.float32(tar_pts)]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    matched_img = cv2.drawMatchesKnn(
        img1=src_img,
        keypoints1=src_pts,
        img2=tar_img,
        keypoints2=tar_pts,
        matches1to2=[matches],
        outImg=None,
        flags=2,
    )
    if border_color is not None:
        matched_img = add_border(matched_img, color=border_color)
    if write_num_matches:
        write_text_on_image(image=matched_img, text=f"{len(matches)} matches")
    if concate_input:
        input_imgs = np.concatenate([src_img, tar_img], axis=1)
        matched_img = np.concatenate([input_imgs, matched_img], axis=0)
    return matched_img


def plot_kpts_pil(img_PIL, points_2d, color="blue"):
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


def draw_contour(img_PIL, mask, color, to_pil=True):
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    img = np.array(img_PIL)
    img[edge, :] = color
    if to_pil:
        return Image.fromarray(img)
    else:
        return img


def overlay_mask_on_rgb(rgb, mask, gray=False, color=(255, 0, 0), alpha=0.5):
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
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((1, 1)))
    img[edge, :] = 255
    return img


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


def make_grid_image(imgs, nrow, padding=5, pad_value=255):
    """
    @imgs: (B,H,W,C) np array
    @nrow: num of images per row
    """
    grid = torchvision.utils.make_grid(
        torch.as_tensor(np.asarray(imgs)).permute(0, 3, 1, 2), nrow=nrow, padding=padding, pad_value=pad_value
    )
    grid = grid.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(np.uint8)
    return grid


def get_cmap(np_img):
    cmap = matplotlib.colormaps.get_cmap("magma")
    tmp = cmap(np_img)[..., :3]
    return tmp
