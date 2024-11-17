import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class VideoDataset(Dataset):
    """Takes in a dataset representing a single video and wraps it to return a seq as a sample rather than a single timestep.

    Args:
        ds: Dataset representing a single video
        seq_len: Number of frames to take for a single sample
        seq_start: Start frame index in the video
        seq_step: Step between frames in a sequence
        num_samples: Number of times to sample a sequence from the video (length of the dataset)
    """

    def __init__(self, ds, seq_len=10, seq_start=None, seq_step=1, num_samples=None, num_workers=8):
        self.ds = ds
        self.seq_len = seq_len
        self.seq_start = seq_start
        self.seq_step = seq_step
        self.num_samples = len(ds) if num_samples is None else num_samples
        self.num_workers = num_workers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sampler = SequenceSampler(len(self.ds), seq_len=self.seq_len, seq_step=self.seq_step, seq_start=self.seq_start)
        dataloader = DataLoader(self.ds, batch_size=1, sampler=sampler, num_workers=self.num_workers)
        seq = []
        for batch in dataloader:
            seq.append(batch)
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

    def __len__(self):
        return sum(len(ds) for ds in self.video_datasets)

    def __getitem__(self, idx):
        video_idx = torch.randint(0, len(self.video_datasets), (1,)).item()
        return self.video_datasets[video_idx][idx]


class SequenceSampler(Sampler):
    def __init__(self, dataset_len, seq_len=None, seq_step=1, seq_start=None):
        self.dataset_len = dataset_len
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.seq_start = seq_start

    def __iter__(self):
        timesteps = min(self.dataset_len, self.seq_len) if self.seq_len is not None else self.dataset_len
        seq_start = self.seq_start

        if seq_start is None:
            seq_start = torch.randint(
                0,
                max(1, self.dataset_len + 1 - timesteps * self.seq_step),
                (1,),
            ).item()

        assert self.seq_step > 0, f"{self.seq_step=}"

        frame_idxs = []
        for t in range(timesteps):
            frame_idx = seq_start + t * self.seq_step
            if frame_idx < self.dataset_len:
                frame_idxs.append(frame_idx)

        return iter(frame_idxs)

    def __len__(self):
        if self.seq_len is not None:
            return self.seq_len
        return self.dataset_len
