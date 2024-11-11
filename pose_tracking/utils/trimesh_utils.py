# https://github.com/nv-nguyen/nope/blob/main/src/utils/trimesh_utils.py
import copy

import cv2
import numpy as np
import scipy
import torch
import trimesh
from PIL import Image
from pose_tracking.config import logger


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=m.vertices, faces=m.faces) for m in scene_or_mesh.geometry.values()]
        )
    else:
        result = scene_or_mesh
    return result


def AABB_to_OBB(AABB):
    """
    AABB bbox to oriented bounding box
    """
    minx, miny, minz, maxx, maxy, maxz = np.arange(6)
    corner_index = np.array(
        [
            minx,
            miny,
            minz,
            maxx,
            miny,
            minz,
            maxx,
            maxy,
            minz,
            minx,
            maxy,
            minz,
            minx,
            miny,
            maxz,
            maxx,
            miny,
            maxz,
            maxx,
            maxy,
            maxz,
            minx,
            maxy,
            maxz,
        ]
    ).reshape((-1, 3))

    corners = AABB.reshape(-1)[corner_index]
    return corners


def load_mesh_bounds(path, ORIGIN_GEOMETRY="BOUNDS", return_origin_bounds=False):
    mesh = as_mesh(trimesh.load(path))
    if ORIGIN_GEOMETRY == "BOUNDS":
        AABB = mesh.bounds
        center = np.mean(AABB, axis=0)
        mesh.vertices -= center
    if return_origin_bounds:
        return mesh, center
    else:
        return mesh


def load_mesh(mesh_path, ext=None):
    if ext is None:
        ext = mesh_path.split(".")[-1]
    mesh = trimesh.load(open(mesh_path, "rb"), file_type=ext, force="mesh")
    bbox = np.asarray(mesh.bounding_box.vertices).copy()
    diameter = compute_mesh_diameter(mesh)
    return {
        "mesh": mesh,
        "bbox": bbox,
        "diameter": diameter,
    }


def get_bbox_from_mesh(mesh):
    AABB = mesh.bounds
    OBB = AABB_to_OBB(AABB)
    return OBB


def get_obj_origin_and_diameter(mesh_path):
    mesh, center = load_mesh_bounds(mesh_path, return_origin_bounds=True)
    extents = mesh.extents * 2
    return np.linalg.norm(extents), center


def extract_data_from_trimesh(mesh, device="cuda", max_tex_size=None):
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert("RGB"))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors["tex"] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        mesh_tensors["uv_idx"] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors["uv"] = uv
    else:
        if mesh.visual.vertex_colors is None:
            logger.info("WARN: mesh doesn't have vertex_colors, assigning a pure color")
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1))
        mesh_tensors["vertex_color"] = (
            torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float) / 255.0
        )

    mesh_tensors.update(
        {
            "pos": torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            "faces": torch.tensor(mesh.faces, device=device, dtype=torch.int),
            "vnormals": torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        }
    )
    return mesh_tensors


def compute_mesh_diameter(mesh):
    u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
    pts = u @ s
    diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return float(diameter)


def add_colored_texture_to_mesh(mesh, color=np.array([255, 255, 255]), resolution=5):
    tex_img = np.tile(color.reshape(1, 1, 3), (resolution, resolution, 1)).astype(np.uint8)
    mesh = mesh.unwrap()
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, image=Image.fromarray(tex_img))
    return mesh


def vis_trimesh(mesh):
    bbox = get_bbox_from_mesh(mesh)
    scene = trimesh.Scene([mesh, trimesh.points.PointCloud(bbox)])
    return scene


def get_posed_model_pts(rt1, model):
    pred_model = copy.deepcopy(model)
    pred_model.apply_transform(rt1)
    pts1 = np.asarray(pred_model.vertices)
    return pts1


if __name__ == "__main__":
    from pose_tracking.config import PROJ_DIR

    mesh_path = PROJ_DIR / "data/ycb/models/021_bleach_cleanser/google_16k/textured.obj"
    mesh = load_mesh(mesh_path)
    scene = vis_trimesh(mesh)
    scene.show()
