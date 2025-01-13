import json
import os
import typing as t
from pathlib import Path

import comet_ml
import cv2
import numpy as np
import torch
import torch.nn as nn
from pose_tracking.utils.comet_utils import (
    log_args,
    log_ckpt_to_exp,
    log_params_to_exp,
    log_pkg_code,
    log_tags,
)
from pose_tracking.utils.common import adjust_img_for_plt, cast_to_numpy
from pose_tracking.utils.misc import DeviceType
from pose_tracking.utils.pose import convert_pose_vector_to_matrix


def save_results(batch_t, pose_pred, preds_dir, gt_pose):
    # batch_t contains data for the t-th timestep in N sequences
    bsize = len(batch_t["rgb"])
    for bidx in range(bsize):
        rgb = cast_to_numpy(batch_t["rgb"][bidx])
        intrinsics = cast_to_numpy(batch_t["intrinsics"][bidx])
        name = Path(batch_t["rgb_path"][bidx]).stem
        pose = torch.eye(4)
        pose = pose_pred[bidx]
        pose = cast_to_numpy(pose)
        gt_pose_formatted = cast_to_numpy(gt_pose[bidx])
        seq_dir = preds_dir if bsize == 1 else preds_dir / f"seq_{bidx}"
        pose_path = seq_dir / "poses" / f"{name}.txt"
        gt_path = seq_dir / "poses_gt" / f"{name}.txt"
        intrinsics_path = seq_dir / "intrinsics.txt"
        rgb_path = seq_dir / "rgb" / f"{name}.png"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_path, "w") as f:
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")
        with open(gt_path, "w") as f:
            for row in gt_pose_formatted:
                f.write(" ".join(map(str, row)) + "\n")
        rgb = adjust_img_for_plt(rgb)
        rgb = rgb[..., ::-1]
        rgb_path = str(rgb_path)
        cv2.imwrite(rgb_path, rgb)
        np.savetxt(intrinsics_path, intrinsics)


def save_results_v2(rgb, intrinsics, pose_gt, pose_pred, rgb_path, preds_dir, bboxs=None, labels=None):
    # batch_t contains data for the t-th timestep in N sequences
    bsize = len(rgb)
    for bidx in range(bsize):
        rgb = cast_to_numpy(rgb[bidx])
        intrinsics = cast_to_numpy(intrinsics[bidx])
        name = Path(rgb_path[bidx]).stem
        pose = torch.eye(4)
        pose = pose_pred[bidx]
        pose = cast_to_numpy(pose)
        gt_pose_formatted = pose_gt[bidx]
        gt_pose_formatted = cast_to_numpy(gt_pose_formatted)
        seq_dir = preds_dir if bsize == 1 else preds_dir / f"seq_{bidx}"
        pose_path = seq_dir / "poses" / f"{name}.txt"
        gt_path = seq_dir / "poses_gt" / f"{name}.txt"
        intrinsics_path = seq_dir / "intrinsics.txt"
        rgb_path = seq_dir / "rgb" / f"{name}.png"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        if bboxs is not None and labels is not None:
            bbox_path = seq_dir / "bbox" / f"{name}.json"
            bbox_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bbox_path, "w") as f:
                json.dump({"bbox": bboxs[bidx].tolist(), "labels": labels[bidx].tolist()}, f)
        with open(pose_path, "w") as f:
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")
        with open(gt_path, "w") as f:
            for row in gt_pose_formatted:
                f.write(" ".join(map(str, row)) + "\n")
        rgb = adjust_img_for_plt(rgb)
        rgb = rgb[..., ::-1]
        rgb_path = str(rgb_path)
        cv2.imwrite(rgb_path, rgb)
        np.savetxt(intrinsics_path, intrinsics)


def load_model_from_ckpt(model, ckpt_path):
    state_dict = torch.load(ckpt_path)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    # rename all occurences of lstm_cell and rnn_cell to state_cell
    for key in list(state_dict.keys()):
        if "lstm_cell" in key or "rnn_cell" in key:
            new_key = key.replace("lstm_cell", "state_cell").replace("rnn_cell", "state_cell")
            state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    return model


def load_from_ckpt(
    checkpoint_path: str,
    device: DeviceType,
    trep_net: nn.Module,
    optimizer: t.Any = None,
    scheduler: t.Any = None,
) -> dict:
    state_dicts = torch.load(
        checkpoint_path,
        map_location=device,
    )
    trep_net.load_state_dict(state_dicts["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = 0.005
    if scheduler is not None:
        scheduler.load_state_dict(state_dicts["scheduler"])

    return {
        "model": trep_net,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def log_artifacts(artifacts: dict, exp: comet_ml.Experiment, log_dir: str, epoch: int, suffix=None) -> str:
    """Logs the training artifacts to the experiment and saves the model and session to the log directory."""

    suffix = epoch if suffix is None else suffix
    save_path_model = os.path.join(log_dir, f"model_{suffix}.pth")
    save_model(artifacts["model"], save_path_model)
    save_path_session = os.path.join(log_dir, "session.pth")
    torch.save(
        {
            "optimizer": artifacts["optimizer"].state_dict(),
            "scheduler": artifacts["scheduler"].state_dict(),
            "epoch": epoch,
        },
        save_path_session,
    )
    log_ckpt_to_exp(exp, save_path_model, "ckpt")
    log_ckpt_to_exp(exp, save_path_session, "ckpt")
    return save_path_model


def save_model(model, save_path_model):
    torch.save(
        {
            "model": model.state_dict(),
        },
        save_path_model,
    )


def log_model_meta(model: nn.Module, exp: comet_ml.Experiment = None, logger=None) -> None:
    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    printer = logger.info if logger is not None else print
    printer(f"{num_params_total=}")

    def sender(x):
        if exp is not None:
            exp.log_parameters(x)

    for name, submodule in model.named_children():  # Only immediate submodules
        num_params = sum(p.numel() for p in submodule.parameters())
        printer(f"num_params_{name}: {num_params}")
        sender({f"model/num_params_{name}": num_params})

    sender({"model/num_params_total": num_params_total})


def log_exp_meta(args, save_args, logdir, exp, args_to_group_map=None):
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    log_tags(args, exp, args_to_group_map=args_to_group_map)
    log_pkg_code(exp)

    if save_args:
        log_args(exp, args, f"{logdir}/args.yaml")
