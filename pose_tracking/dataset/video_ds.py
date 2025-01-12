import torch
from pose_tracking.dataset.dataloading import preload_ds
from pose_tracking.dataset.ds_common import adjust_img_for_torch
from pose_tracking.dataset.transforms import (
    apply_replay_transform,
    get_transforms_video,
)
from pose_tracking.utils.common import adjust_img_for_plt
from pose_tracking.utils.misc import print_cls
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
    ):
        self.do_preload = do_preload

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
                max(1, len(self.ds) + 1 - timesteps * seq_step),
                (1,),
            ).item()

        assert seq_step > 0, f"{seq_step=}"

        for t in range(timesteps):
            frame_idx = seq_start + t * seq_step
            if self.do_preload:
                sample = self.seq[frame_idx]
            else:
                sample = self.ds[frame_idx]
            if sample is None:
                return None
            if self.transforms_rgb is not None:
                if t == 0:
                    t_res = self.transforms_rgb(image=adjust_img_for_plt(sample["rgb"]))
                else:
                    t_res = apply_replay_transform(adjust_img_for_plt(sample["rgb"]), t_res)
                new_sample = {k: v for k, v in sample.items() if k not in ["rgb"]}
                new_sample["rgb"] = adjust_img_for_torch(t_res["image"])
                sample = new_sample
            seq.append(sample)
        return seq


class VideoDatasetTracking(VideoDataset):
    """Trackformer data adapter
    """

    def __getitem__(self, idx):
        seq = []
        timesteps = min(self.seq_len, len(self.ds) - 1)
        seq_start = self.seq_start
        seq_step = self.seq_step

        if not seq_step:
            seq_step = torch.randint(1, self.max_random_seq_step, (1,)).item()

        if seq_start is None:
            seq_start = torch.randint(
                1,
                max(1, len(self.ds) + 1 - timesteps * seq_step),
                (1,),
            ).item()
        else:
            seq_start = max(1, seq_start)
            timesteps = min(timesteps, (len(self.ds) - seq_start) // seq_step)

        assert seq_step > 0, f"{seq_step=}"

        for t in range(timesteps):
            frame_idx = seq_start + t * seq_step
            frame_idx_prev = frame_idx - seq_step
            sample = self.ds[frame_idx]
            if sample is None:
                return None
            sample_prev = self.ds[frame_idx_prev]
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

            for k in ["boxes", "prev_boxes", "rot", "t", "prev_rot", "prev_t"]:
                new_sample[k] = new_sample[k].float()
                new_sample[k] = new_sample[k][None] if new_sample[k].ndim == 1 else new_sample[k]

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
                if k.startswith("prev_") and k not in ["prev_image"]
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
                ]
            }

            new_sample["target"]["prev_target"] = new_sample.pop("prev_target")

            sample = new_sample
            seq.append(sample)
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
