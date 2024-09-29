"""Based on https://pytorch.org/blog/understanding-gpu-memory-1/"""

import os
import pickle
from pathlib import Path

import torch


def start_record_memory_history() -> None:
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not recording memory history")
        return

    print("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        enabled=True, trace_alloc_max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not recording memory history")
        return

    print("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=False)


def export_memory_snapshot(path="memory_snapshot.pkl"):
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not exporting memory snapshot")
        return

    try:
        print(f"Saving snapshot to local file: {path}")
        snapshot = torch.cuda.memory._snapshot()
        pickle.dump(snapshot, open(path, "wb"))
        return snapshot
    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")
        return


def oom_observer(device, alloc, device_alloc, device_free):
    # snapshot right after an OOM happened
    print("saving allocated state during OOM")
    snapshot = torch.cuda.memory._snapshot()
    pickle.dump(snapshot, open("oom_snapshot.pickle", "wb"))


def attach_oom_observer():
    torch._C._cuda_attach_out_of_memory_observer(oom_observer)


def profile_func(func, save_path, *args, **kwargs):
    """Profile a function and save the results to a file."""
    from cProfile import Profile
    from pstats import SortKey, Stats

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with Profile() as profile:
            func(*args, **kwargs)
    except KeyboardInterrupt:
        print("Caught the KeyboardInterrupt, still saving profile")
    finally:
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.TIME)
            .dump_stats(save_path)
        )
        msg = f"snakeviz {save_path}"
        try:
            import pyperclip
            pyperclip.copy(msg)
        except ImportError:
            pass
        print(msg)


if __name__ == "__main__":
    """Using a profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
            run_code()

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    """

    attach_oom_observer()
    start_record_memory_history()
    # do some work
    stop_record_memory_history()
    export_memory_snapshot()
