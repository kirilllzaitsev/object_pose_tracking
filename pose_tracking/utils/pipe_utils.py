import argparse
import os
import typing as t
from datetime import datetime
from pathlib import Path

import comet_ml
import torch
import torch.nn as nn
from pose_tracking.config import ARTIFACTS_DIR, PROJ_NAME
from pose_tracking.models.cnnlstm import RecurrentCNN
from pose_tracking.utils.comet_utils import (
    create_tracking_exp,
    log_args,
    log_ckpt_to_exp,
    log_params_to_exp,
    log_tags,
)
from pose_tracking.utils.misc import DeviceType
from torch.utils.tensorboard import SummaryWriter


def create_tools(args: argparse.Namespace) -> dict:
    """Creates tools for the pipeline:
    - Comet experiment
    - writer
    - logdir
    Logs the arguments and tags to the experiment.
    """
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = f"{ARTIFACTS_DIR}/{args.exp_name}/{now}"
    os.makedirs(logdir, exist_ok=True)
    preds_base_dir = f"{logdir}/preds"
    preds_dir = Path(preds_base_dir)
    exp = create_tracking_exp(args, project_name=PROJ_NAME)
    args.comet_exp_name = exp.name  # automatically assigned by Comet

    writer = SummaryWriter(log_dir=logdir, flush_secs=10)
    return {
        "exp": exp,
        "writer": writer,
        "preds_dir": preds_dir,
        "logdir": logdir,
    }


def log_exp_meta(args, save_args, logdir, exp, args_to_group_map=None):
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    log_tags(args, exp, args_to_group_map=args_to_group_map)

    if save_args:
        log_args(exp, args, f"{logdir}/args.yaml")


def load_model_from_ckpt(model, ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
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
    save_path_model = os.path.join(log_dir, f"model_{suffix}.pt")
    log_model(artifacts, save_path_model)
    save_path_session = os.path.join(log_dir, "session.pt")
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


def log_model(model, save_path_model):
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


def print_stats(train_stats, logger, stage):
    logger.info(f"## {stage.upper()} ##")
    LOSS_METRICS = ["loss", "loss_pose", "loss_depth", "loss_rot", "loss_trans"]
    ERROR_METRICS = ["r_err", "t_err"]
    ADDITIONAL_METRICS = ["add", "adds", "miou", "5deg5cm", "2deg2cm"]

    for stat_group in [LOSS_METRICS, ERROR_METRICS, ADDITIONAL_METRICS]:
        msg = []
        for k in stat_group:
            if k in train_stats:
                msg.append(f"{k}: {train_stats[k]:.4f}")
        logger.info(" | ".join(msg))


class Printer:
    def __init__(self, logger):
        self.logger = logger

    def saved_artifacts(self, epoch):
        self.logger.info(f"Saved artifacts on epoch {epoch}")
