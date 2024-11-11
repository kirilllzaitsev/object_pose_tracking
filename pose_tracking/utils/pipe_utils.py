import argparse
from datetime import datetime
from pathlib import Path

from pose_tracking.config import ARTIFACTS_DIR
from pose_tracking.utils.comet_utils import (
    create_tracking_exp,
    log_args,
    log_params_to_exp,
    log_tags,
)
from torch.utils.tensorboard import SummaryWriter


def create_tools(args: argparse.Namespace, save_args: bool = True) -> dict:
    """Creates tools for the pipeline:
    - Comet experiment
    - writer
    - logdir
    Logs the arguments and tags to the experiment.
    """
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = f"{ARTIFACTS_DIR}/{args.exp_name}_{now}"
    preds_base_dir = f"{logdir}/preds"
    preds_dir = Path(preds_base_dir) / now
    exp = create_tracking_exp(args, project_name="pose_tracking")
    args.run_name = exp.name  # automatically assigned by Comet
    log_params_to_exp(
        exp,
        vars(args),
        "args",
    )
    log_tags(args, exp)

    if save_args:
        log_args(exp, args, f"{logdir}/args.yaml")

    writer = SummaryWriter(log_dir=logdir, flush_secs=10)
    return {
        "exp": exp,
        "writer": writer,
        "preds_dir": preds_dir,
        "logdir": logdir,
    }
