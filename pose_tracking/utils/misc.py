import random
import typing as t

import numpy as np
import torch

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


def to_numpy(x: TensorOrArrOrList) -> np.ndarray:
    if isinstance(x, list):
        return np.array([to_numpy(xx) for xx in x])
    elif isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()
