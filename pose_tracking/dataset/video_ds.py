import torch
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

    def __init__(self, ds, seq_len=10, seq_start=None, seq_step=1, num_samples=None):
        self.ds = ds
        self.seq_len = min(len(self.ds), seq_len) if seq_len is not None else len(self.ds)
        self.seq_start = seq_start
        self.seq_step = seq_step
        self.num_samples = len(ds) if num_samples is None else num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = []
        timesteps = self.seq_len
        seq_start = self.seq_start
        if seq_start is None:
            seq_start = torch.randint(
                0,
                max(1, len(self.ds) + 1 - timesteps * self.seq_step),
                (1,),
            ).item()
        assert self.seq_step > 0, f"{self.seq_step=}"

        for t in range(timesteps):
            frame_idx = seq_start + t * self.seq_step
            sample = self.ds[frame_idx]
            seq.append(sample)
        return seq


class VideoDatasetPreload(VideoDataset):
    """Preloads the entire dataset in memory for faster access."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq = super().__getitem__(0)

    def __getitem__(self, idx):
        return self.seq


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
        ds_idx = torch.randint(0, len(self.video_datasets), (1,)).item()
        sample_idx = idx - sum(self.lens[:ds_idx])
        return self.video_datasets[ds_idx][sample_idx]
