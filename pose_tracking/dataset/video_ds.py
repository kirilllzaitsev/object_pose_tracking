import torch
from pose_tracking.dataset.dataloading import preload_ds
from pose_tracking.dataset.ds_common import adjust_img_for_torch
from pose_tracking.dataset.transforms import (
    apply_replay_transform,
    get_transforms_video,
)
from pose_tracking.utils.common import adjust_img_for_plt
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
        self, ds, seq_len=None, seq_start=0, seq_step=1, num_samples=None, do_preload=False, transforms_rgb=None
    ):
        self.do_preload = do_preload

        self.ds = ds
        self.seq_start = seq_start
        self.seq_step = seq_step

        self.seq_len = min(len(self.ds), seq_len) if seq_len is not None else len(self.ds)
        self.num_samples = len(ds) if num_samples is None else num_samples
        self.transforms_rgb = get_transforms_video(transforms_rgb) if transforms_rgb is not None else None

        if do_preload:
            self.seq = preload_ds(ds)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = []
        timesteps = self.seq_len
        seq_start = self.seq_start
        seq_step = self.seq_step

        if not seq_step:
            seq_step = torch.randint(1, 7, (1,)).item()

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
