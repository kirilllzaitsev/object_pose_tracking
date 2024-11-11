import argparse
import os
from datetime import datetime
from pathlib import Path

from pose_tracking.config import ARTIFACTS_DIR, PROJ_NAME
from pose_tracking.utils.comet_utils import (
    create_tracking_exp,
    log_args,
    log_params_to_exp,
    log_tags,
)
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
    args.run_name = exp.name  # automatically assigned by Comet

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
