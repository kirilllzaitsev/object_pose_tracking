import torch
from pose_tracking.utils.rotation_conversions import matrix_to_quaternion


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene, transform=None, obj_idx=0):
        self.transform = transform
        self.num_samples = len(scene["frame_id"])
        self.scene = scene
        self.obj_idx = obj_idx

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame_id = self.scene["frame_id"][idx]
        rgb = self.scene["rgb"][idx]
        mask = self.scene["mask"][idx][self.obj_idx]
        rgb_path = self.scene["rgb_path"][idx]

        if self.transform:
            rgb = self.transform(rgb)
            mask = self.transform(mask)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        pose = self.scene["pose"][idx][self.obj_idx]
        r = pose[:3, :3]
        r_quat = matrix_to_quaternion(r)
        t = pose[:3, 3] / 1e3

        return {"frame_id": frame_id, "rgb": rgb, "mask": mask, "r": r_quat, "t": t, "rgb_path": rgb_path}

    def clone(self, idxs=None):
        scene = self.scene
        if idxs is not None:
            scene = {k: [v[i] for i in idxs] for k, v in scene.items()}
        return SceneDataset(scene, self.transform)
