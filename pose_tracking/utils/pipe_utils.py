import argparse
import os
import typing as t
from datetime import datetime
from pathlib import Path

import comet_ml
import cv2
import torch
import torch.nn as nn
from pose_tracking.config import ARTIFACTS_DIR, PROJ_NAME, YCBINEOAT_SCENE_DIR
from pose_tracking.dataset.custom_sim_ds import CustomSimDataset
from pose_tracking.dataset.video_ds import MultiVideoDataset, VideoDataset
from pose_tracking.dataset.ycbineoat import YCBineoatDataset
from pose_tracking.losses import compute_add_loss, geodesic_loss
from pose_tracking.models.cnnlstm import RecurrentCNN
from pose_tracking.utils.comet_utils import (
    create_tracking_exp,
    log_args,
    log_ckpt_to_exp,
    log_params_to_exp,
    log_tags,
)
from pose_tracking.utils.common import adjust_img_for_plt, cast_to_numpy
from pose_tracking.utils.misc import DeviceType
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from torch.utils.tensorboard import SummaryWriter


def get_model(args):
    priv_dim = 256 * 3
    latent_dim = 256  # defined by the encoders
    depth_dim = latent_dim
    rgb_dim = latent_dim
    model = RecurrentCNN(
        depth_dim=depth_dim,
        rgb_dim=rgb_dim,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        bdec_priv_decoder_out_dim=priv_dim,
        bdec_priv_decoder_hidden_dim=args.bdec_priv_decoder_hidden_dim,
        bdec_depth_decoder_hidden_dim=args.bdec_depth_decoder_hidden_dim,
        benc_belief_enc_hidden_dim=args.benc_belief_enc_hidden_dim,
        benc_belief_depth_enc_hidden_dim=args.benc_belief_depth_enc_hidden_dim,
        bdec_hidden_attn_hidden_dim=args.bdec_hidden_attn_hidden_dim,
        encoder_name=args.encoder_name,
        do_predict_2d=args.do_predict_2d,
        do_predict_6d_rot=args.do_predict_6d_rot,
        benc_belief_enc_num_layers=args.benc_belief_enc_num_layers,
        benc_belief_depth_enc_num_layers=args.benc_belief_depth_enc_num_layers,
        priv_decoder_num_layers=args.priv_decoder_num_layers,
        depth_decoder_num_layers=args.depth_decoder_num_layers,
        hidden_attn_num_layers=args.hidden_attn_num_layers,
        rt_mlps_num_layers=args.rt_mlps_num_layers,
        dropout=args.dropout,
        use_rnn=not args.no_rnn,
        use_obs_belief=not args.no_obs_belief,
        use_priv_decoder=args.use_priv_decoder,
        do_freeze_encoders=args.do_freeze_encoders,
        use_prev_pose_condition=args.use_prev_pose_condition,
    )

    return model


def get_trainer(args, model, device, writer=None, world_size=1):
    from pose_tracking.train import Trainer

    criterion_trans = nn.MSELoss()
    criterion_rot = geodesic_loss
    use_pose_loss = args.pose_loss_name in ["add"]
    assert not (use_pose_loss and args.do_predict_2d), "tmp:pose loss implemented only for direct 3d"
    criterion_pose = compute_add_loss if use_pose_loss else None

    trainer = Trainer(
        model=model,
        device=device,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        criterion_trans=criterion_trans,
        criterion_rot=criterion_rot,
        criterion_pose=criterion_pose,
        writer=writer,
        do_predict_2d=args.do_predict_2d,
        do_predict_6d_rot=args.do_predict_6d_rot,
        use_rnn=not args.no_rnn,
        use_obs_belief=not args.no_obs_belief,
        world_size=world_size,
        do_log_every_ts=args.do_log_every_ts,
        do_log_every_seq=args.do_log_every_seq,
        use_ddp=args.use_ddp,
        use_prev_pose_condition=args.use_prev_pose_condition,
    )

    return trainer


def create_tools(args: argparse.Namespace) -> dict:
    """Creates tools for the pipeline:
    - Comet experiment
    - writer
    - logdir
    Logs the arguments and tags to the experiment.
    """
    exp = create_tracking_exp(args, project_name=PROJ_NAME)
    args.comet_exp_name = exp.name  # automatically assigned by Comet
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_name = args.comet_exp_name or f"{args.exp_name}/{now}"
    logdir = f"{ARTIFACTS_DIR}/{exp_name}"
    os.makedirs(logdir, exist_ok=True)
    preds_base_dir = f"{logdir}/preds"
    preds_dir = Path(preds_base_dir)

    writer = SummaryWriter(log_dir=logdir, flush_secs=10)
    return {
        "exp": exp,
        "writer": writer,
        "preds_dir": preds_dir,
        "logdir": logdir,
    }


def save_results(batch_t, t_pred, rot_pred, preds_dir):
    # batch_t contains data for the t-th timestep in N sequences
    batch_size = len(batch_t["rgb"])
    for seq_idx in range(batch_size):
        rgb = batch_t["rgb"][seq_idx].cpu().numpy()
        name = Path(batch_t["rgb_path"][seq_idx]).stem
        pose = torch.eye(4)
        r_quat = rot_pred[seq_idx]
        pose[:3, :3] = quaternion_to_matrix(r_quat) if r_quat.shape != (3, 3) else r_quat
        pose[:3, 3] = t_pred[seq_idx]
        pose = cast_to_numpy(pose)
        gt_pose = batch_t["pose"][seq_idx]
        gt_pose_formatted = convert_pose_quaternion_to_matrix(gt_pose)
        gt_pose_formatted[:3, 3] = gt_pose[:3].squeeze()
        gt_pose_formatted = cast_to_numpy(gt_pose_formatted)
        seq_dir = preds_dir if batch_size == 1 else preds_dir / f"seq_{seq_idx}"
        pose_path = seq_dir / "poses" / f"{name}.txt"
        gt_path = seq_dir / "poses_gt" / f"{name}.txt"
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


def reduce_metric(value, world_size):
    """Synchronize and average a metric across all processes."""
    tensor = torch.tensor(value, device="cuda")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item() / world_size


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
    log_model(artifacts["model"], save_path_model)
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


def get_full_ds(
    obj_names,
    ds_name,
    seq_len,
    seq_step,
    seq_start,
    num_samples,
    ds_kwargs,
):
    video_datasets = []
    for obj_name in obj_names:
        ds = get_obj_ds(ds_name, ds_kwargs, obj_name)
        video_ds = VideoDataset(
            ds=ds,
            seq_len=seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            num_samples=num_samples,
        )
        video_datasets.append(video_ds)

    if len(video_datasets) > 1:
        full_ds = MultiVideoDataset(video_datasets)
    else:
        full_ds = video_datasets[0]
    return full_ds


def get_obj_ds(ds_name, ds_kwargs, obj_name):
    if ds_name == "ycbi":
        ds = YCBineoatDataset(video_dir=YCBINEOAT_SCENE_DIR / obj_name, **ds_kwargs)
    elif ds_name == "cube_sim":
        ds = CustomSimDataset(
            obj_name=obj_name,
            **ds_kwargs,
        )
    else:
        raise NotImplementedError(f"Unknown dataset name {ds_name}")
    return ds


def print_stats(train_stats, logger, stage):
    logger.info(f"## {stage.upper()} ##")
    LOSS_METRICS = ["loss", "loss_pose", "loss_depth", "loss_rot", "loss_t", "loss_priv"]
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
