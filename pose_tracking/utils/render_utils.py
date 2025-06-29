import threading

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from pose_tracking.utils.misc import add_batch_dim_to_img, is_empty, print_cls
from pose_tracking.utils.pose import convert_pose_vector_to_matrix
from pose_tracking.utils.trimesh_utils import load_mesh
from torchvision.transforms.functional import rgb_to_grayscale

glcam_in_cvcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(float)


def adjust_brightness(src_rgb, target_rgb):
    src_rgb = add_batch_dim_to_img(src_rgb)
    target_rgb = add_batch_dim_to_img(target_rgb)

    gray1 = rgb_to_grayscale(src_rgb)
    gray2 = rgb_to_grayscale(target_rgb)
    mask1 = gray1 > 0
    mask2 = gray2 > 0

    mean1 = torch.sum(gray1 * mask1, dim=(1, 2, 3)) / (mask1.sum(dim=(1, 2, 3)) + 1e-8)
    mean2 = torch.sum(gray2 * mask2, dim=(1, 2, 3)) / (mask2.sum(dim=(1, 2, 3)) + 1e-8)
    mean2 = mean2.to(mean1.device)

    scale = (mean2 / (mean1 + 1e-8)).view(-1, 1, 1, 1)
    max_val = 1.0 if src_rgb.max() <= 1.0 else 255.0

    return torch.clamp(src_rgb * scale, 0, max_val)


def transform_pts(pts, tf):
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def transform_dirs(dirs, tf):
    """
    @dirs: (...,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]


def make_mesh_tensors(mesh, device="cuda", max_tex_size=None):
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
            # logging.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
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


def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords="y_down"):
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == "y_up":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == "y_down":
        proj = np.array(
            [
                [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
                [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
                [0, 0, q, qn],  # Sets near and far planes (glPerspective).
                [0, 0, -1, 0],
            ]
        )
    else:
        raise NotImplementedError

    return proj


def to_homo_torch(pts):
    """
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    """
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones), dim=-1)
    return homo


def init_nvdiffrast():
    import nvdiffrast.torch as dr

    print("Pre-building interpolate kernel...")
    ctx = dr.RasterizeGLContext(device=torch.device("cuda:0"))  # create temp context
    dummy_pos = torch.zeros(1, 3, 4, device="cuda:0")
    dummy_tri = torch.zeros(1, 3, dtype=torch.int32, device="cuda:0")
    dummy_rast, _ = dr.rasterize(ctx, dummy_pos, dummy_tri, [1, 1])

    dummy_attr = torch.zeros(1, 3, 4, device="cuda:0")
    _ = dr.interpolate(dummy_attr, dummy_rast, dummy_tri)
    print("Interpolate kernel built.")


class Dispatcher:
    # https://github.com/NVlabs/nvdiffrast/issues/23
    def __init__(self, gpu_ids):
        self.threads = {}
        self.events = {}
        self.funcs = {}
        self.return_events = {}
        self.return_values = {}

        for gpu_id in gpu_ids:
            device = torch.device(gpu_id)
            self.events[device] = threading.Event()
            self.return_events[device] = threading.Event()
            self.threads[device] = threading.Thread(
                target=Dispatcher.worker,
                args=(
                    self,
                    device,
                ),
                daemon=True,
            )
            self.threads[device].start()

    @staticmethod
    def worker(self, device):
        torch.cuda.set_device(device)
        ctx = dr.RasterizeCudaContext(device=device)
        while True:
            self.events[device].wait()
            assert device not in self.return_values
            self.return_values[device] = self.funcs[device](ctx)
            del self.funcs[device]
            self.events[device].clear()
            self.return_events[device].set()

    def __call__(self, device, func):
        assert device not in self.funcs
        self.funcs[device] = func
        self.events[device].set()
        self.return_events[device].wait()
        ret_val = self.return_values[device]
        del self.return_values[device]
        self.return_events[device].clear()
        return ret_val

    def __repr__(self) -> str:
        return print_cls(self)


class Rasterizer(torch.nn.Module):
    def __init__(self, dispatcher, device):
        super().__init__()
        self.dispatcher = dispatcher
        self.device = device

    def forward(self, pos, tri, resolution, mesh_tensors, pts_cam):
        try:

            def func(ctx):
                has_tex = "tex" in mesh_tensors
                rast_out, _ = dr.rasterize(ctx, pos=pos, tri=tri, resolution=resolution)
                xyz_map, _ = dr.interpolate(pts_cam, rast_out, tri)
                device = pos.device
                depth = xyz_map[..., 2]
                if has_tex:
                    texc, _ = dr.interpolate(mesh_tensors["uv"].to(device), rast_out, mesh_tensors["uv_idx"].to(device))
                    color = dr.texture(mesh_tensors["tex"].to(device), texc, filter_mode="linear")
                else:
                    color, _ = dr.interpolate(mesh_tensors["vertex_color"].to(device), rast_out, tri)
                return {
                    "depth": depth,
                    "color": color,
                    "rast_out": rast_out,
                    "xyz_map": xyz_map,
                }

            return self.dispatcher(pos.device, func)
        except Exception as e:
            print(locals())
            print(f"{e=}")
            print(self.dispatcher)
            raise e


def nvdiffrast_render(
    glctx,
    K=None,
    H=None,
    W=None,
    ob_in_cams=None,
    get_normal=False,
    mesh_tensors=None,
    mesh=None,
    projection_mat=None,
    bbox2d=None,
    output_size=None,
    use_light=False,
    light_color=None,
    light_dir=np.array([0, 0, 1]),
    light_pos=np.array([0, 0, 0]),
    w_ambient=0.8,
    w_diffuse=0.5,
    extra={},
    rasterize_fn=None,
):
    """Just plain rendering, not support any gradient
    @K: (3,3) np array
    @ob_in_cams: (N,4,4) torch tensor, openCV camera
    @projection_mat: np array (4,4)
    @output_size: (height, width)
    @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
    @light_dir: in cam space
    @light_pos: in cam space
    """

    if mesh_tensors is None:
        mesh_tensors = make_mesh_tensors(mesh)
    pos = mesh_tensors["pos"]
    vnormals = mesh_tensors["vnormals"]
    pos_idx = mesh_tensors["faces"]
    has_tex = "tex" in mesh_tensors
    device = K.device

    ob_in_glcams = torch.tensor(glcam_in_cvcam, device=device, dtype=torch.float)[None] @ ob_in_cams
    if projection_mat is None:
        projection_mat = torch.tensor(
            projection_matrix_from_intrinsics(K.cpu(), height=H, width=W, znear=0.1, zfar=100)
        ).cuda()
    projection_mat = torch.as_tensor(projection_mat.reshape(-1, 4, 4), device=device, dtype=torch.float)
    mtx = projection_mat @ ob_in_glcams

    if output_size is None:
        output_size = np.asarray([H, W])

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    if bbox2d is not None:
        l = bbox2d[:, 0]
        t = H - bbox2d[:, 1]
        r = bbox2d[:, 2]
        b = H - bbox2d[:, 3]
        tf = torch.eye(4, dtype=torch.float, device=device).reshape(1, 4, 4).expand(len(ob_in_cams), 4, 4).contiguous()
        tf[:, 0, 0] = W / (r - l)
        tf[:, 1, 1] = H / (t - b)
        tf[:, 3, 0] = (W - r - l) / (r - l)
        tf[:, 3, 1] = (H - t - b) / (t - b)
        pos_clip = pos_clip @ tf

    if rasterize_fn is None:
        rast_out, _ = dr.rasterize(glctx=glctx, pos=pos_clip, tri=pos_idx, resolution=np.asarray(output_size))
        xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
        depth = xyz_map[..., 2]
        if has_tex:
            texc, _ = dr.interpolate(mesh_tensors["uv"], rast_out, mesh_tensors["uv_idx"])
            color = dr.texture(mesh_tensors["tex"], texc, filter_mode="linear")
        else:
            color, _ = dr.interpolate(mesh_tensors["vertex_color"], rast_out, pos_idx)
    else:
        r_res = rasterize_fn(
            pos=pos_clip, tri=pos_idx, resolution=np.asarray(output_size), mesh_tensors=mesh_tensors, pts_cam=pts_cam
        )
        rast_out = r_res["rast_out"]
        xyz_map = r_res["xyz_map"]
        depth = r_res["depth"]
        color = r_res["color"]

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device="cuda")
        else:
            light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device="cuda").reshape(1, 1, 3) - pts_cam
        diffuse_intensity = (
            (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[..., None]
        )
        diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
        if light_color is None:
            light_color = color
        else:
            light_color = torch.as_tensor(light_color, device="cuda", dtype=torch.float)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

    color = color.clip(0, 1)
    mask = torch.clamp(rast_out[..., -1:], 0, 1)

    color = color * mask  # Mask out background using alpha
    color = torch.flip(color, dims=[1])  # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    extra["xyz_map"] = torch.flip(xyz_map, dims=[1])
    mask = torch.flip(mask, dims=[1])
    return {
        "color": color,
        "depth": depth,
        "normal": normal_map,
        "mask": mask,
    }


def render_batch_pose_preds(batch, poses_pred, glctx=None, rasterize_fn=None, use_light=False):
    K = batch["intrinsics"]
    h, w = batch["rgb"].shape[-2:]
    mesh = batch["mesh"]
    poses_pred = convert_pose_vector_to_matrix(poses_pred)
    assert poses_pred.ndim >= 3, poses_pred.shape
    assert poses_pred.device.type == "cuda", poses_pred.device.type
    bs = poses_pred.shape[0]
    input_resize = h, w
    rgb_rs = []
    depth_rs = []
    normal_rs = []
    xyz_map_rs = []
    mask_rs = []
    for bidx in range(bs):
        if isinstance(mesh, trimesh.Trimesh):
            mesh_bidx = mesh
        else:
            mesh_bidx = mesh[bidx]
            assert len(mesh_bidx) == 1, len(mesh_bidx)  # only one obj for now
            mesh_bidx = mesh_bidx[0]

        mesh_tensors = make_mesh_tensors(mesh_bidx, device=poses_pred.device)

        def get_r_res(pose):
            extra = {}
            r_res = nvdiffrast_render(
                K=K[bidx].to(poses_pred.device),
                H=h,
                W=w,
                ob_in_cams=pose.to(poses_pred.device),
                get_normal=False,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                output_size=input_resize,
                # bbox2d=bbox2d_ori[b : b + bs],
                use_light=use_light,  # shouldn't matter for feature extraction
                extra=extra,
                rasterize_fn=rasterize_fn,
            )
            rgb_r, depth_r, normal_r, mask_r = (
                r_res["color"],
                r_res["depth"],
                r_res["normal"],
                r_res["mask"],
            )
            rgb_rs.append(rgb_r)
            depth_rs.append(depth_r[..., None])
            normal_rs.append(normal_r)
            xyz_map_rs.append(extra["xyz_map"])
            mask_rs.append(mask_r)
            return r_res

        if poses_pred.ndim == 4:
            for qidx in range(poses_pred.shape[1]):
                get_r_res(poses_pred[bidx, qidx : qidx + 1])
        else:
            get_r_res(poses_pred[bidx : bidx + 1])

    return {
        "rgb": torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2),
        "depth": torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2) if not is_empty(depth_rs) else None,
        "normal": torch.cat(normal_rs, dim=0).permute(0, 3, 1, 2) if not is_empty(normal_rs) else None,
        "xyz_map": torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2) if not is_empty(xyz_map_rs) else None,
        "mask": torch.cat(mask_rs, dim=0).permute(0, 3, 1, 2) if not is_empty(mask_rs) else None,
    }
