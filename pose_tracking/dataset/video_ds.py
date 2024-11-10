import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """Takes in a dataset representing a single video and wraps it to return a seq as a sample rather than a single timestep.

    Args:
        ds: Dataset representing a single video
        seq_len: Number of frames to take for a single sample
        seq_start: Start frame index in the video
        seq_step: Step between frames in a sequence
        num_samples: Number of times to sample a sequence from the video
    """

    def __init__(self, ds, seq_len=10, seq_start=0, seq_step=1, num_samples=None):
        self.ds = ds
        self.seq_len = seq_len
        self.seq_start = seq_start
        self.seq_step = seq_step
        self.num_samples = len(ds) if num_samples is None else num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = []
        timesteps = self.seq_len if self.seq_len is not None else len(self.ds)
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
