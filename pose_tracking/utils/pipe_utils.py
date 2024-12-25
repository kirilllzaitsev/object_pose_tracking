import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from pose_tracking.config import ARTIFACTS_DIR, PROJ_NAME, RELATED_DIR, YCB_MESHES_DIR
from pose_tracking.dataset.custom_sim_ds import (
    CustomSimDatasetCube,
    CustomSimDatasetIkea,
)
from pose_tracking.dataset.transforms import get_transforms
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
        backbone = build_backbone(detr_args)

        transformer = build_deforamble_transformer(detr_args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.mt_num_queries,
            num_feature_levels=detr_args.num_feature_levels,
            aux_loss=detr_args.aux_loss,
            with_box_refine=detr_args.with_box_refine,
            two_stage=detr_args.two_stage,
        )
    elif args.model_name == "detr_basic":
        from pose_tracking.models.detr import DETR

        model = DETR(
            num_classes=num_classes,
            n_queries=args.mt_num_queries,
            d_model=args.mt_d_model,
            n_tokens=args.mt_n_tokens,
            n_layers=args.mt_n_layers,
            n_heads=args.mt_n_heads,
        )
    elif args.model_name == "detr_kpt":
        from pose_tracking.models.detr import KeypointDETR

        model = KeypointDETR(
            num_classes=num_classes,
            n_queries=args.mt_num_queries,
            kpt_spatial_dim=args.mt_kpt_spatial_dim,
            encoding_type=args.mt_encoding_type,
            d_model=args.mt_d_model,
            n_tokens=args.mt_n_tokens,
            n_layers=args.mt_n_layers,
            n_heads=args.mt_n_heads,
        )
    else:
        num_pts = 256
        priv_dim = num_pts * 3
        latent_dim = args.encoder_out_dim  # defined by the encoders
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
            encoder_depth_weights=args.encoder_depth_weights,
            encoder_img_weights=args.encoder_img_weights,
            norm_layer_type=args.norm_layer_type,
            encoder_out_dim=args.encoder_out_dim,
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
                    "kpt_spatial_dim": args.mt_kpt_spatial_dim,
                    "do_calibrate_kpt": args.mt_do_calibrate_kpt,
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
        opt_only=args.opt_only,
        criterion_rot_name=args.rot_loss_name,
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
    logdir = Path(f"{ARTIFACTS_DIR}/{exp_name}")
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


def get_datasets(
    ds_name,
    seq_len,
    seq_step,
    seq_start,
    ds_video_dir_train=None,
    ds_video_dir_val=None,
    ds_video_dir_test=None,
    ds_video_subdirs_train=None,
    ds_video_subdirs_val=None,
    ds_video_subdirs_test=None,
    transform_names=None,
    transform_prob=0.0,
    mask_pixels_prob=0.0,
    num_samples=None,
    ds_types=("train", "val"),
    model_name="",
    do_predict_kpts=False,
    use_priv_decoder=False,
    do_overfit=False,
    do_preload_ds=False,
    include_mask=False,
    do_convert_pose_to_quat=True,
):

    transform_rgb = get_transforms(transform_names, transform_prob=transform_prob) if transform_names else None
    is_detr_model = "detr" in model_name
    is_roi_model = "_sep" in model_name
    is_pizza_model = "pizza" in model_name
    include_bbox_2d = do_predict_kpts or is_detr_model or is_roi_model or is_pizza_model
    ds_kwargs_common = dict(
        shorter_side=None,
        zfar=np.inf,
        include_mask=include_mask or include_bbox_2d,  # mask is needed for 2d bbox
        include_bbox_2d=include_bbox_2d,
        start_frame_idx=0,
        do_convert_pose_to_quat=do_convert_pose_to_quat,
        mask_pixels_prob=mask_pixels_prob,
        do_normalize_bbox=True if is_detr_model else False,
        bbox_format="cxcywh" if is_detr_model else "xyxy",
        model_name="pizza" if is_pizza_model else model_name,
    )
    if ds_name == "ycbi":
        ycbi_kwargs = dict(
            ycb_meshes_dir=YCB_MESHES_DIR,
        )
        ds_kwargs_custom = ycbi_kwargs
    elif ds_name == "ikea":
        ikea_kwargs = dict()
        ds_kwargs_custom = ikea_kwargs
    else:
        cube_sim_kwargs = dict(
            mesh_path=f"{ds_video_dir_train}/mesh/cube.obj",
            use_priv_info=use_priv_decoder,
        )
        ds_kwargs_custom = cube_sim_kwargs

    ds_kwargs = {**ds_kwargs_common, **ds_kwargs_custom}

    res = {}

    train_ds_kwargs = copy.deepcopy(ds_kwargs)
    train_ds_kwargs["video_dir"] = ds_video_dir_train
    if "train" in ds_types:
        train_dataset = get_video_ds(
            ds_video_subdirs=ds_video_subdirs_train,
            ds_name=ds_name,
            seq_len=seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            ds_kwargs=train_ds_kwargs,
            num_samples=num_samples,
            do_preload=do_preload_ds,
            transforms_rgb=transform_rgb,
        )
        mesh_paths_orig_train = [d.ds.mesh_path_orig for d in train_dataset.video_datasets]
        res["train"] = train_dataset
    else:
        mesh_paths_orig_train = None

    if "val" in ds_types:
        if do_overfit:
            val_dataset = get_video_ds(
                ds_video_subdirs=ds_video_subdirs_train,
                ds_name=ds_name,
                seq_len=None,
                seq_step=1,
                seq_start=seq_start,
                ds_kwargs=train_ds_kwargs,
                num_samples=None,
                do_preload=do_preload_ds,
                transforms_rgb=None,
            )
        else:
            assert ds_video_dir_val and ds_video_subdirs_val
            val_ds_kwargs = copy.deepcopy(ds_kwargs)
            val_ds_kwargs.pop("mask_pixels_prob")
            val_ds_kwargs["video_dir"] = ds_video_dir_val
            val_dataset = get_video_ds(
                ds_video_subdirs=ds_video_subdirs_val,
                ds_name=ds_name,
                seq_len=seq_len if is_pizza_model else None,
                seq_step=seq_step if is_pizza_model else 1,
                seq_start=0,
                ds_kwargs=val_ds_kwargs,
                num_samples=num_samples,
                do_preload=True,
                mesh_paths_to_take=mesh_paths_orig_train,
            )
        res["val"] = val_dataset

    if "test" in ds_types:
        assert ds_video_dir_test and ds_video_subdirs_test
        test_ds_kwargs = copy.deepcopy(ds_kwargs)
        test_ds_kwargs.pop("mask_pixels_prob")
        test_ds_kwargs["video_dir"] = ds_video_dir_test
        test_dataset = get_video_ds(
            ds_video_subdirs=ds_video_subdirs_test,
            ds_name=ds_name,
            seq_len=seq_len,
            seq_step=1,
            seq_start=0,
            ds_kwargs=test_ds_kwargs,
            num_samples=num_samples,
            do_preload=True,
        )
        res["test"] = test_dataset

    return res


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
    mesh_paths_to_take=None,
):
    video_datasets = []
    for ds_video_subdir in tqdm(ds_video_subdirs, leave=False, desc="Video datasets"):
        ds = get_obj_ds(ds_name, ds_kwargs, ds_video_subdir=ds_video_subdir)
        seq_len = len(ds) if seq_len is None else seq_len
        if mesh_paths_to_take is not None:
            if ds.mesh_path_orig not in mesh_paths_to_take:
                continue
        video_ds = VideoDataset(
            ds=ds,
            seq_len=seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            num_samples=num_samples,
            do_preload=do_preload,
            transforms_rgb=transforms_rgb,
        )
        video_datasets.append(video_ds)

    full_ds = MultiVideoDataset(video_datasets)
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
