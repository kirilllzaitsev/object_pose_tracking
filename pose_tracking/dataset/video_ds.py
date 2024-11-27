import functools
from multiprocessing.pool import ThreadPool

import torch
from torch.utils.data import Dataset


def load_sample(i, ds):
    return ds[i]


class VideoDataset(Dataset):
    """Takes in a dataset representing a single video and wraps it to return a seq as a sample rather than a single timestep.

    Args:
        ds: Dataset representing a single video
        seq_len: Number of frames to take for a single sample
        seq_start: Start frame index in the video
        seq_step: Step between frames in a sequence
        num_samples: Number of times to sample a sequence from the video (length of the dataset)
    """

    def __init__(self, ds, seq_len=10, seq_start=None, seq_step=1, num_samples=None, do_preload=False):
        self.do_preload = do_preload

        self.ds = ds
        self.seq_start = seq_start
        self.seq_step = seq_step

        self.seq_len = min(len(self.ds), seq_len) if seq_len is not None else len(self.ds)
        self.num_samples = len(ds) if num_samples is None else num_samples

        if do_preload:
            with ThreadPool() as pool:
                self.seq = pool.map(functools.partial(load_sample, ds=self.ds), range(len(self.ds)))

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
        ds_idx = torch.randint(0, len(self.video_datasets), (1,)).item()
        sample_idx = idx - sum(self.lens[:ds_idx])
        return self.video_datasets[ds_idx][sample_idx]
