import functools
import typing as t
from multiprocessing.pool import ThreadPool

from pose_tracking.utils.misc import DeviceType, is_tensor, to


def transfer_batch_to_device(batch: t.Union[dict, list], device):
    if isinstance(batch, dict):
        batch = _transfer_batch_to_device(batch, device)
    elif isinstance(batch, list):
        batch = [_transfer_batch_to_device(b, device) for b in batch]
    return batch


def _transfer_batch_to_device(batch: dict, device: DeviceType) -> dict:
    """Transfer the batch of data to the device. The data can be a dictionary, list, or a tensor."""
    for k, v in batch.items():
        if is_tensor(v):
            batch[k] = to(v, device)
        elif isinstance(v, list):
            if len(v) == 0:
                continue
            if is_tensor(v[0]):
                batch[k] = [to(x, device) for x in v]
            elif isinstance(v[0], list):
                batch[k] = [[_transfer_batch_to_device(x, device) for x in y] for y in v]
            elif isinstance(v[0], dict):
                batch[k] = [_transfer_batch_to_device(x, device) for x in v]
        elif isinstance(v, dict):
            batch[k] = _transfer_batch_to_device(v, device)
    return batch


def load_sample(i, ds):
    return ds[i]


def preload_ds(ds):
    with ThreadPool() as pool:
        seq = pool.map(functools.partial(load_sample, ds=ds), range(len(ds)))
    return seq
