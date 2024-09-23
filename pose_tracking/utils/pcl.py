import typing as t

import numpy as np
import open3d as o3d
import torch

TensorOrArr = t.Union[torch.Tensor, np.ndarray]


def downsample_pcl_via_voxels(xyz: TensorOrArr, voxel_size: float) -> TensorOrArr:
    """Downsample point cloud by averaging points within every voxel of size voxel_size."""
    pcd = o3d.geometry.PointCloud()
    is_tensor = isinstance(xyz, torch.Tensor)
    if is_tensor:
        device = xyz.device
        xyz = xyz.detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    new_pts = np.asarray(downpcd.points)
    if is_tensor:
        new_pts = torch.from_numpy(new_pts).to(device)
    return new_pts
