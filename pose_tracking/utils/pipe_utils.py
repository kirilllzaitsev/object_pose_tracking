import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml
from pose_tracking.config import ARTIFACTS_DIR, PROJ_NAME, RELATED_DIR
from pose_tracking.dataset.custom_sim_ds import (
    CustomSimDatasetCube,
    CustomSimDatasetIkea,
)
from pose_tracking.dataset.video_ds import MultiVideoDataset, VideoDataset
from pose_tracking.dataset.ycbineoat import YCBineoatDataset, YCBineoatDatasetPizza
from pose_tracking.losses import compute_add_loss, get_rot_loss, get_t_loss
from pose_tracking.models.cnnlstm import RecurrentCNN, RecurrentCNNSeparated
from pose_tracking.trainer import TrainerPizza
from pose_tracking.utils.comet_utils import create_tracking_exp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_model(args):
    num_classes = getattr(args, "num_classes", 21)

    if args.model_name == "videopose":
        from videopose.arguments import parse_args
        from videopose.models.model import VideoPose

        args = parse_args()
        # args.backbone = "transformer"
        model = VideoPose(args)
    elif args.model_name == "pizza":
        from pizza.lib.model.network import PIZZA

        model = PIZZA(backbone="resnet18", img_feature_dim=512, multi_frame=False).cuda()
    elif args.model_name == "detr":
        from deformable_detr.models.backbone import build_backbone
        from deformable_detr.models.deformable_detr import DeformableDETR
        from deformable_detr.models.deformable_transformer import (
            build_deforamble_transformer,
        )

        detr_args = yaml.load(
            open(
                f"{RELATED_DIR}/transformers/Deformable-DETR/deformable_detr/config.yaml",
                "r",
            ),
            Loader=yaml.Loader,
        )
        detr_args.device = args.device
        detr_args.num_queries = 10
        backbone = build_backbone(detr_args)

        transformer = build_deforamble_transformer(detr_args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=detr_args.num_queries,
            num_feature_levels=detr_args.num_feature_levels,
            aux_loss=detr_args.aux_loss,
            with_box_refine=detr_args.with_box_refine,
            two_stage=detr_args.two_stage,
        )
    elif args.model_name == "detr_basic":
        from pose_tracking.models.detr import DETR

        model = DETR(
            num_classes=num_classes,
            n_queries=args.num_queries,
            d_model=args.d_model,
            n_tokens=args.n_tokens,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    elif args.model_name == "detr_kpt":
        from pose_tracking.models.detr import KeypointDETR

        model = KeypointDETR(
            num_classes=num_classes,
            n_queries=args.num_queries,
            kpt_spatial_dim=args.kpt_spatial_dim,
            encoding_type=args.encoding_type,
            d_model=args.d_model,
            n_tokens=args.n_tokens,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        )
    elif args.model_name == "detr":
        from deformable_detr.models.backbone import build_backbone
        from deformable_detr.models.deformable_detr import DeformableDETR
        from deformable_detr.models.deformable_transformer import (
            build_deforamble_transformer,
        )

        detr_args = yaml.load(
            open(
                f"{RELATED_DIR}/transformers/Deformable-DETR/deformable_detr/config.yaml",
                "r",
            ),
            Loader=yaml.Loader,
        )
        detr_args.device = args.device
        backbone = build_backbone(detr_args)

        transformer = build_deforamble_transformer(detr_args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=detr_args.num_queries,
            num_feature_levels=detr_args.num_feature_levels,
            aux_loss=detr_args.aux_loss,
            with_box_refine=detr_args.with_box_refine,
            two_stage=detr_args.two_stage,
        )
    elif args.model_name == "detr_basic":
        from pose_tracking.models.detr import DETR

        model = DETR(num_classes=num_classes)
    else:
        priv_dim = 256 * 3
        latent_dim = 256  # defined by the encoders
        depth_dim = latent_dim
        rgb_dim = latent_dim
        if args.model_name == "cnnlstm":
            model_cls = RecurrentCNN
        elif args.model_name == "cnnlstm_sep":
            model_cls = RecurrentCNNSeparated
        else:
            raise ValueError(f"Unknown model name {args.model_name}")
        model = model_cls(
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
            do_predict_2d_t=args.do_predict_2d_t,
            do_predict_6d_rot=args.do_predict_6d_rot,
            do_predict_3d_rot=args.do_predict_3d_rot,
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
            use_prev_latent=args.use_prev_latent,
            do_predict_kpts=args.do_predict_kpts,
        )

    return model


def get_trainer(args, model, device, writer=None, world_size=1, logger=None, do_vis=False, exp_dir=None):
    from pose_tracking.trainer import Trainer, TrainerDeformableDETR, TrainerVideopose

    criterion_trans = get_t_loss(args.t_loss_name)
    criterion_rot = get_rot_loss(args.rot_loss_name)
    use_pose_loss = args.pose_loss_name in ["add"]
    criterion_pose = compute_add_loss if use_pose_loss else None

    if args.model_name in ["videopose"]:
        trainer_cls = TrainerVideopose
    elif args.model_name in ["pizza"]:
        trainer_cls = TrainerPizza
    elif "detr" in args.model_name:
        trainer_cls = TrainerDeformableDETR
    else:
        trainer_cls = Trainer

    if "detr" in args.model_name:
        extra_kwargs = {
            "num_classes": 21,
        }
        if "detr" in args.model_name:
            extra_kwargs.update(
                {
                    "num_dec_layers": 6,
                    "aux_loss": True,
                }
            )
        if "detr_kpt" in args.model_name:
            extra_kwargs.update(
                {
                    "kpt_spatial_dim": args.kpt_spatial_dim,
                    "do_calibrate_kpt": args.do_calibrate_kpt,
                }
            )
    else:
        extra_kwargs = {}

    trainer = trainer_cls(
        model=model,
        device=device,
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        criterion_trans=criterion_trans,
        criterion_rot=criterion_rot,
        criterion_pose=criterion_pose,
        writer=writer,
        seq_len=args.seq_len,
        do_predict_2d_t=args.do_predict_2d_t,
        do_predict_6d_rot=args.do_predict_6d_rot,
        do_predict_3d_rot=args.do_predict_3d_rot,
        use_rnn=not args.no_rnn,
        use_obs_belief=not args.no_obs_belief,
        world_size=world_size,
        logger=logger,
        vis_epoch_freq=args.vis_epoch_freq,
        exp_dir=exp_dir,
        do_log_every_ts=args.do_log_every_ts,
        do_log_every_seq=args.do_log_every_seq,
        use_ddp=args.use_ddp,
        use_prev_pose_condition=args.use_prev_pose_condition,
        do_predict_rel_pose=args.do_predict_rel_pose,
        do_predict_kpts=args.do_predict_kpts,
        use_prev_latent=args.use_prev_latent,
        do_vis=do_vis,
        model_name=args.model_name,
        do_debug=args.do_debug,
        do_print_seq_stats=args.do_print_seq_stats,
        **extra_kwargs,
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
    if args.comet_exp_name:
        exp_name = args.comet_exp_name
    else:
        exp_name = f"{args.exp_name}/{now}"
        if len(args.exp_tags) > 0:
            exp_name = f"{exp_name}_{'_'.join(args.exp_tags)}"
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


def get_video_ds(
    ds_video_subdirs,
    ds_name,
    seq_len,
    seq_step,
    seq_start,
    ds_kwargs,
    num_samples=None,
    do_preload=False,
    transforms_rgb=None,
):
    video_datasets = []
    for ds_video_subdir in tqdm(ds_video_subdirs, leave=False, desc="Video datasets"):
        ds = get_obj_ds(ds_name, ds_kwargs, ds_video_subdir=ds_video_subdir)
        seq_len = len(ds) if seq_len is None else seq_len
        video_ds = VideoDataset(
            ds=ds,
            seq_len=seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            num_samples=max(1, len(ds) // seq_len) if num_samples is None else num_samples,
            do_preload=do_preload,
            transforms_rgb=transforms_rgb,
        )
        video_datasets.append(video_ds)

    if len(video_datasets) > 1:
        full_ds = MultiVideoDataset(video_datasets)
    else:
        full_ds = video_datasets[0]
    return full_ds


def get_obj_ds(ds_name, ds_kwargs, ds_video_subdir):
    ds_kwargs = ds_kwargs.copy()
    video_dir = Path(ds_kwargs.pop("video_dir"))
    model_name = ds_kwargs.pop("model_name", None)
    if ds_name == "ycbi":
        if model_name == "pizza":
            ds_cls = YCBineoatDataset
        else:
            ds_cls = YCBineoatDataset
        ds = ds_cls(video_dir=video_dir / ds_video_subdir, **ds_kwargs)
    elif ds_name == "cube_sim":
        ds = CustomSimDatasetCube(
            **ds_kwargs,
        )
    elif ds_name == "ikea":
        ds = CustomSimDatasetIkea(
            video_dir=video_dir / ds_video_subdir,
            **ds_kwargs,
        )
    else:
        raise NotImplementedError(f"Unknown dataset name {ds_name}")
    return ds


class Printer:
    def __init__(self, logger=None):
        self.log_fn = logger.info if logger else print

    def saved_artifacts(self, epoch):
        self.log_fn(f"Saved artifacts on epoch {epoch}")

    def print_stats(self, train_stats, stage):
        self.log_fn(f"## {stage.upper()} ##")
        LOSS_METRICS = [k for k in train_stats if "loss" in k]
        ERROR_METRICS = [k for k in train_stats if "err" in k]
        ADDITIONAL_METRICS = ["add", "adds", "miou", "5deg5cm", "2deg2cm", "nan_count"]

        for stat_group in [LOSS_METRICS, ERROR_METRICS, ADDITIONAL_METRICS]:
            msg = []
            for k in stat_group:
                if k in train_stats:
                    msg.append(f"{k}: {train_stats[k]:.4f}")
            self.log_fn(" | ".join(msg))
