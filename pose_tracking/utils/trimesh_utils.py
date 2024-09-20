# https://github.com/nv-nguyen/nope/blob/main/src/utils/trimesh_utils.py
import numpy as np
import trimesh


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


def load_mesh(path, ORIGIN_GEOMETRY="BOUNDS", return_origin_bounds=False):
    mesh = as_mesh(trimesh.load(path))
    if ORIGIN_GEOMETRY == "BOUNDS":
        AABB = mesh.bounds
        center = np.mean(AABB, axis=0)
        mesh.vertices -= center
    if return_origin_bounds:
        return mesh, center
    else:
        return mesh


def get_bbox_from_mesh(mesh):
    AABB = mesh.bounds
    OBB = AABB_to_OBB(AABB)
    return OBB


def get_obj_origin_and_diameter(mesh_path):
    mesh, center = load_mesh(mesh_path, return_origin_bounds=True)
    extents = mesh.extents * 2
    return np.linalg.norm(extents), center


def vis_trimesh(mesh):
    bbox = get_bbox_from_mesh(mesh)
    scene = trimesh.Scene([mesh, trimesh.points.PointCloud(bbox)])
    scene.show()


if __name__ == "__main__":
    from pose_tracking.config import PROJ_DIR

    mesh_path = PROJ_DIR / "data/ycb/models/021_bleach_cleanser/google_16k/textured.obj"
    mesh = load_mesh(mesh_path)
    vis_trimesh(mesh)
