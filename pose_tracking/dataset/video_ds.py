import functools

import torch
from pose_tracking.dataset.dataloading import preload_ds
from pose_tracking.dataset.ds_common import adjust_img_for_torch
from pose_tracking.dataset.transforms import (
    apply_replay_transform,
    get_transforms_video,
)
from pose_tracking.utils.common import adjust_img_for_plt
from pose_tracking.utils.geom import convert_3d_t_for_2d, pose_to_egocentric_delta_pose
from pose_tracking.utils.misc import print_cls
from pose_tracking.utils.pose import (
    convert_pose_matrix_to_vector,
    convert_pose_vector_to_matrix,
)
from pose_tracking.utils.rotation_conversions import convert_rotation_representation
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Takes in a dataset representing a single video and wraps it to return a seq as a sample rather than a single timestep.

    Args:
        ds: Dataset representing a single video
        seq_len: Number of frames to take for a single sample
        seq_start: Start frame index in the video
        seq_step: Step between frames in a sequence
        num_samples: Number of times to sample a sequence from the video (length of the dataset)
    """

    def __init__(
        self,
        ds,
        seq_len=None,
        seq_start=None,
        seq_step=1,
        num_samples=None,
        do_preload=False,
        transforms_rgb=None,
        max_random_seq_step=8,
        do_predict_rel_pose=False,
    ):
        self.do_preload = do_preload
        self.do_predict_rel_pose = do_predict_rel_pose

        self.ds = ds
        self.seq_start = seq_start
        self.seq_step = seq_step
        self.max_random_seq_step = max_random_seq_step

        self.seq_len = min(len(self.ds), seq_len) if seq_len is not None else len(self.ds)
        if seq_start is not None:
            self.num_samples = 1 if self.seq_step else self.max_random_seq_step - 1  # because of the fixed start idx
        else:
            # not very meaningful. can sample much more different subsequences given random seq_step/seq_start
            self.num_samples = max(1, len(ds) // self.seq_len) if num_samples is None else num_samples
        self.transforms_rgb = get_transforms_video(transforms_rgb) if transforms_rgb is not None else None

        if do_preload:
            self.seq = preload_ds(ds)

    def __len__(self):
        return self.num_samples

    def __repr__(self) -> str:
        return print_cls(self, excluded_attrs=["seq"])

    def __getitem__(self, idx):
        seq = []
        timesteps = self.seq_len
        seq_start = self.seq_start
        seq_step = self.seq_step

        if not seq_step:
            seq_step = torch.randint(1, self.max_random_seq_step, (1,)).item()

        if seq_start is None:
            seq_start = torch.randint(
                0,
                max(1, len(self.ds) - 1 - (timesteps - 1) * seq_step),
                (1,),
            ).item()

        assert seq_step > 0, f"{seq_step=}"

        idxs = [seq_start + t * seq_step for t in range(timesteps)]
        if self.do_preload:
            seq = [self.seq[idx] for idx in idxs]
        elif timesteps > 10:
            seq = preload_ds(self.ds, idxs=idxs)
        else:
            for t in range(timesteps):
                sample = self.ds[idxs[t]]
                seq.append(sample)

        for t, sample in enumerate(seq):
            if sample is None:
                return None
            if self.transforms_rgb is not None:
                if t == 0:
                    t_res = self.transforms_rgb(image=adjust_img_for_plt(sample["rgb"]))
                else:
                    t_res = apply_replay_transform(adjust_img_for_plt(sample["rgb"]), t_res)
                new_sample = {k: v for k, v in sample.items() if k not in ["rgb"]}
                new_sample["rgb"] = adjust_img_for_torch(t_res["image"])
                seq[t] = new_sample

        return seq


class VideoDatasetTracking(VideoDataset):
    """Trackformer data adapter"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.t_repr = self.ds.t_repr
        self.rot_repr = self.ds.rot_repr
        self.pose_to_mat_converter_fn = functools.partial(convert_pose_vector_to_matrix, rot_repr=self.rot_repr)
        self.pose_mat_to_vector_converter_fn = functools.partial(convert_pose_matrix_to_vector, rot_repr=self.rot_repr)

    def __getitem__(self, idx):
        timesteps = min(self.seq_len, len(self.ds) - 1)
        seq_start = self.seq_start
        seq_step = self.seq_step

        if not seq_step:
            seq_step = torch.randint(1, self.max_random_seq_step, (1,)).item()

        if seq_start is None:
            seq_start = torch.randint(
                seq_step,
                max(seq_step, len(self.ds) - 1 - (timesteps - 1) * seq_step),
                (1,),
            ).item()
        else:
            seq_start = max(1, seq_start)
            timesteps = min(timesteps, (len(self.ds) - seq_start) // seq_step)
        seq_start = max(seq_start, seq_step)

        assert seq_step > 0, f"{seq_step=}"

        idxs = [seq_start - seq_step] + [seq_start + t * seq_step for t in range(timesteps)]

        if self.do_preload:
            seq = [self.seq[idx] for idx in idxs]
        elif timesteps > 10:
            seq = preload_ds(self.ds, idxs=idxs)
        else:
            seq = [self.ds[idx] for idx in idxs]

        for idx in range(1, len(seq)):
            sample = seq[idx]
            sample_prev = seq[idx - 1]
            if sample is None:
                return None
            if sample_prev is None:
                return None
            new_sample = {k: v for k, v in sample.items() if k not in ["rgb"]}
            new_sample["rgb"] = adjust_img_for_torch(sample["rgb"])
            # rename rgb->image bbox_2d->boxes class_id->labels
            new_sample["bbox_2d"] = sample["bbox_2d"]
            new_sample["class_id"] = sample["class_id"]
            new_sample["intrinsics"] = sample["intrinsics"]
            new_sample["size"] = torch.tensor(sample["rgb"].shape[-2:])
            pose = sample["pose"]
            new_sample["pose"] = pose
            # pose is always in (3d t, rot) format
            new_sample["t"] = pose[..., :3]
            new_sample["rot"] = pose[..., 3:]

            new_sample["prev_rgb"] = adjust_img_for_torch(sample_prev["rgb"])
            new_sample["prev_bbox_2d"] = sample_prev["bbox_2d"]
            new_sample["prev_class_id"] = sample_prev["class_id"]
            new_sample["prev_intrinsics"] = sample_prev["intrinsics"]
            new_sample["prev_size"] = torch.tensor(sample_prev["rgb"].shape[-2:])

            prev_pose = sample_prev["pose"]
            new_sample["prev_pose"] = prev_pose
            new_sample["prev_t"] = prev_pose[..., :3]
            new_sample["prev_rot"] = prev_pose[..., 3:]

            new_sample["image"] = new_sample.pop("rgb")
            new_sample["prev_image"] = new_sample.pop("prev_rgb")
            new_sample["boxes"] = new_sample.pop("bbox_2d")
            new_sample["labels"] = new_sample.pop("class_id")
            new_sample["prev_boxes"] = new_sample.pop("prev_bbox_2d")
            new_sample["prev_labels"] = new_sample.pop("prev_class_id")
            new_sample["prev_rgb_path"] = sample_prev["rgb_path"]

            if self.do_predict_rel_pose:
                pose_mat_prev_gt_abs = self.pose_to_mat_converter_fn(prev_pose)
                pose_mat_gt_abs = self.pose_to_mat_converter_fn(pose)
                if pose_mat_prev_gt_abs.ndim == 2:
                    pose_mat_prev_gt_abs = pose_mat_prev_gt_abs[None]
                    pose_mat_gt_abs = pose_mat_gt_abs[None]

                t_gt_rel, rot_gt_rel_mat = pose_to_egocentric_delta_pose(pose_mat_prev_gt_abs, pose_mat_gt_abs)
                rot_gt_rel = convert_rotation_representation(rot_gt_rel_mat, rot_representation=self.ds.rot_repr)

                if self.t_repr == "2d":
                    prev_t_gt_2d, _ = convert_3d_t_for_2d(
                        new_sample["prev_t"][None],
                        intrinsics=new_sample["prev_intrinsics"][None],
                        hw=new_sample["prev_size"],
                    )
                    t_gt_2d, _ = convert_3d_t_for_2d(
                        new_sample["t"][None], intrinsics=new_sample["intrinsics"][None], hw=new_sample["size"]
                    )
                    new_sample["xy_rel"] = t_gt_2d - prev_t_gt_2d
                    new_sample["center_depth_rel"] = t_gt_rel[..., 2:]
                else:
                    new_sample["t_rel"] = t_gt_rel
                new_sample["pose_rel"] = torch.cat([t_gt_rel, rot_gt_rel], dim=-1)
                new_sample["rot_rel"] = rot_gt_rel

            for k in ["boxes", "prev_boxes", "rot", "t", "prev_rot", "prev_t", "xy", "center_depth"]:
                if k not in new_sample:
                    continue
                new_sample[k] = new_sample[k].float()
                new_sample[k] = new_sample[k][None] if new_sample[k].ndim == 1 else new_sample[k]

            frame_idx = idxs[idx]
            frame_idx_prev = idxs[idx - 1]
            new_sample["image_id"] = torch.tensor([frame_idx])
            new_sample["track_ids"] = torch.tensor(
                [0]
            )  # always 0 with one obj per video. should persist across frames in this video
            new_sample["prev_image_id"] = torch.tensor([frame_idx_prev])
            new_sample["prev_track_ids"] = torch.tensor([0])

            # move prev_* into prev_target dict removing prev_ prefix
            new_sample["prev_target"] = {
                k.replace("prev_", ""): v
                for k, v in new_sample.items()
                if k.startswith("prev_") and k not in ["prev_image", "prev_rgb_path"]
            }
            for k in new_sample["prev_target"]:
                new_sample.pop(f"prev_{k}")

            new_sample["target"] = {
                k: new_sample.pop(k)
                for k in [
                    "image_id",
                    "track_ids",
                    "boxes",
                    "labels",
                    "intrinsics",
                    "pose",
                    "size",
                    "prev_image",
                    "rot",
                    "t",
                    "xy",
                    "center_depth",
                    "center_depth_rel",
                    "xy_rel",
                    "t_rel",
                    "pose_rel",
                    "rot_rel",
                ]
                if k in new_sample
            }

            new_sample["target"]["prev_target"] = new_sample.pop("prev_target")

            seq[idx - 1] = new_sample

        seq = seq[:-1]  # the last one does not have a pair

        return seq


class MultiVideoDataset(Dataset):
    """
    Takes in multiple datasets representing different videos and wraps them to return a seq from a random video.

    Args:
        video_datasets: List of datasets representing different videos
    """

    def __init__(self, video_datasets: list[VideoDataset]):
        self.video_datasets = video_datasets
        self.lens = [len(ds) for ds in self.video_datasets]

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        for dataset_idx, dataset_len in enumerate(self.lens):
            if idx < dataset_len:
                return self.video_datasets[dataset_idx][idx]
            idx -= dataset_len

        raise IndexError(f"Index {idx} out of range for {len(self)=}")
