import random
import typing as t

import numpy as np
import torch
import torch.distributed as dist

TensorOrArr = t.Union[torch.Tensor, np.ndarray]
TensorOrArrOrList = t.Union[list, torch.Tensor, np.ndarray]
DeviceType = t.Union[str, torch.device]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_env():
    import os

    if os.path.exists("/cluster"):
        return "cluster"
    else:
        return "local"


def pick_library(x: TensorOrArr) -> t.Any:
    if isinstance(x, torch.Tensor):
        lib = torch
    else:
        lib = np
    return lib


def to(x: torch.Tensor, device: DeviceType) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def is_tensor(v):
    return isinstance(v, torch.Tensor)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_metric(value, world_size):
    """Synchronize and average a metric across all processes."""
    tensor = torch.tensor(value, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
    )
    return total_norm


class NestedTensor(object):
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def print_cls(cls, exclude_private=True, excluded_attrs=None, extra_str=None):
    msg = ""
    excluded_attrs = excluded_attrs or []
    for k, v in cls.__dict__.items():
        if exclude_private and k.startswith("_"):
            continue
        if k in excluded_attrs:
            continue
        msg += f"{k}: {v}\n"
    if extra_str:
        msg += f"\n{extra_str}"
    return msg


def split_arr(arr, num_chunks):
    k, m = divmod(len(arr), num_chunks)
    return [arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]
