import concurrent
import gc
import random
import sys
import typing as t

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

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
    if isinstance(x, list):
        return [to(xx, device) for xx in x]
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
    if world_size < 2:
        return value
    if not is_tensor(value):
        tensor = torch.tensor(value, device="cuda")
    else:
        tensor = value
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size


def reduce_dict(input_dict, average=True, device=None):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2 or is_empty(input_dict):
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            v = input_dict[k]
            if not is_tensor(v):
                v = torch.tensor(v, device=device)
            values.append(v)
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
    msg = "\n" + "-" * 30 + "\n"
    msg += f"self: {type(cls)}\n"
    excluded_attrs = excluded_attrs or []
    attrs = cls.__dict__.items()
    attrs = sorted(attrs, key=lambda x: x[0])
    for k, v in attrs:
        if exclude_private and k.startswith("_"):
            continue
        if k in excluded_attrs:
            continue
        msg += f"{k}: {v}\n"
    if len(excluded_attrs) > 0:
        msg += f"\nAlso contains: {excluded_attrs}"
    if extra_str:
        msg += f"\n{extra_str}"
    msg += "\n" + "-" * 30
    return msg


class ClsPrinter:
    def __repr__(self) -> str:
        return print_cls(self)


def split_arr(arr, num_chunks):
    k, m = divmod(len(arr), num_chunks)
    return [arr[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(num_chunks)]


def match_module_by_name(n, name_keywords):
    out = False
    if isinstance(name_keywords, str):
        name_keywords = [name_keywords]
    for b in name_keywords:
        if b in n.split(".") or ("." in b and b in n):
            out = True
            break
    return out


def free_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()


def is_empty(v):
    # returns true if no values are in the dict/list. recursive.
    if isinstance(v, dict):
        return all(is_empty(x) for x in v.values())
    elif isinstance(v, list) and len(v) > 0:
        return all(is_empty(x) for x in v)
    elif is_tensor(v):
        return v.ndim > 0 and len(v) == 0
    elif hasattr(v, "__len__"):
        return len(v) == 0
    else:
        return False


def init_params(module, excluded_names=None, included_names=None):
    excluded_names = [] if excluded_names is None else excluded_names
    included_names = [] if included_names is None else included_names
    for n, m in module.named_modules():
        not_included = len(included_names) > 0 and not any([match_module_by_name(n, x) for x in included_names])
        is_excluded = any([match_module_by_name(n, x) for x in excluded_names])
        if is_excluded or not_included:
            continue
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def print_error_locals():
    # Get exception info: type, value, and traceback
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is None:
        return

    # Walk to the last traceback frame where the exception occurred.
    tb_last = exc_tb
    while tb_last.tb_next is not None:
        tb_last = tb_last.tb_next

    frame = tb_last.tb_frame
    print("=== Exception occurred in function: {} ===".format(frame.f_code.co_name))
    print("Local variables at the point of error:")
    for var, val in frame.f_locals.items():
        try:
            print(f"  {var} = {val}")
        except Exception as e:
            print(f"  {var} = <unprintable: {e}>")
    print("========================================")


def wrap_with_futures(arr, func, use_threads=True, max_workers=None, disable_tqdm=False):
    if use_threads:
        executor_cls = concurrent.futures.ThreadPoolExecutor
    else:
        executor_cls = concurrent.futures.ProcessPoolExecutor
    disable_tqdm = disable_tqdm or len(arr) <= 5
    with executor_cls(max_workers=max_workers) as executor:
        res = list(tqdm(executor.map(func, arr), total=len(arr), disable=disable_tqdm))
        res = [x for x in res if x is not None]
    return res


def get_scale_factor(mesh_path):
    from pose_tracking.utils.trimesh_utils import load_mesh

    scale_factor = None
    if any(x in str(mesh_path) for x in ["allegro", "dextreme"]):
        bbox = load_mesh(mesh_path)["bbox"]
        if abs(bbox[0, 0]) == 1.0:
            scale_factor = 0.0325
        elif abs(bbox[0, 0]) == 0.299:
            scale_factor = 1.08
    return scale_factor


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def add_batch_dim_to_img(img):
    if img.ndim == 3:
        img = img[None]
    return img
