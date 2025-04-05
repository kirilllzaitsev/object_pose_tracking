import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from pose_tracking.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    HO3D_ROOT,
    MEMOTR_DIR,
    NOCS_SCENE_DIR,
    PROJ_DIR,
    PROJ_NAME,
    RELATED_DIR,
    TF_DIR,
    YCB_MESHES_DIR,
    YCBINEOAT_SCENE_DIR,
    YCBV_SCENE_DIR,
)
from pose_tracking.dataset.custom import CustomDataset, CustomDatasetTest
from pose_tracking.dataset.custom_sim_ds import (
    CustomSimDatasetCube,
    CustomSimDatasetIkea,
    CustomSimMultiObjDatasetIkea,
)
from pose_tracking.dataset.ho3d import HO3DDataset
from pose_tracking.dataset.transforms import get_transforms
from pose_tracking.dataset.video_ds import (
    MultiVideoDataset,
    VideoDataset,
    VideoDatasetTracking,
)
from pose_tracking.dataset.ycbineoat import YCBineoatDataset, YCBineoatDatasetPizza
from pose_tracking.dataset.ycbv_ds import YCBvDataset
from pose_tracking.losses import compute_add_loss, get_rot_loss, get_t_loss
from pose_tracking.models.baselines import (
    CNN,
    KeypointCNN,
    KeypointKeypoint,
    KeypointPose,
)
from pose_tracking.models.cnnlstm import (
    RecurrentCNN,
    RecurrentCNNDouble,
    RecurrentCNNSeparated,
    RecurrentCNNVanilla,
)
from pose_tracking.models.cvae import CVAE
from pose_tracking.models.pizza import PIZZA, PizzaWrapper
from pose_tracking.utils.artifact_utils import load_from_ckpt, load_model_from_exp
from pose_tracking.utils.comet_utils import create_tracking_exp
from pose_tracking.utils.common import get_ordered_paths
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("trimesh").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.ERROR)


def get_model(args, num_classes=None):
    # num_classes excluding bg
    if args.model_name == "videopose":
        from videopose.arguments import parse_args
        from videopose.models.model import VideoPose

        args = parse_args()
        # args.backbone = "transformer"
        model = VideoPose(args)
    elif args.model_name == "pizza":

        model_pizza = PIZZA(
            backbone=args.encoder_name,
            img_feature_dim=args.encoder_out_dim,
            # multi_frame=False,
            multi_frame=True,
        )

        model = PizzaWrapper(model_pizza)
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
        detr_args.dropout = args.dropout
        detr_args.num_queries = args.mt_num_queries
        detr_args.enc_layers = args.mt_n_layers
        detr_args.dec_layers = args.mt_n_layers
        # detr_args.dim_feedforward = args.hidden_dim
        detr_args.hidden_dim = args.hidden_dim
        detr_args.nheads = args.mt_n_heads
        args.detr_args = detr_args
        backbone = build_backbone(detr_args)

        transformer = build_deforamble_transformer(detr_args)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes + 1,
            num_queries=args.mt_num_queries,
            num_feature_levels=detr_args.num_feature_levels,
            aux_loss=detr_args.aux_loss,
            with_box_refine=detr_args.with_box_refine,
            two_stage=detr_args.two_stage,
            rot_out_dim=args.rot_out_dim,
            t_out_dim=args.t_out_dim,
        )
    elif args.model_name == "memotr":
        from memotr.models import build_model

        memotr_args = get_memotr_args(args)
        args.memotr_args = memotr_args
        model = build_model(memotr_args, num_classes=num_classes)

        for p in model.parameters():
            p.requires_grad = True
    elif args.model_name == "trackformer":
        from trackformer.models import build_model

        detr_args = get_trackformer_args(args)
        args.detr_args = detr_args
        model, *_ = build_model(detr_args, num_classes=num_classes + 1)

        if args.tf_use_pretrained_model:
            # mind bbox embed. should have 3 layers
            assert args.hidden_dim == 288, args.hidden_dim
            obj_detect_checkpoint = torch.load(
                f"{TF_DIR}/models/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_hidden_dim_288.pth",
                map_location="cpu",
            )

            obj_detect_state_dict = obj_detect_checkpoint["model"]

            obj_detect_state_dict = {
                k.replace("detr.", ""): v
                for k, v in obj_detect_state_dict.items()
                if "track_encoding" not in k and "class_embed" not in k
            }

            model.load_state_dict(obj_detect_state_dict, strict=False)

            if hasattr(model, "tracking"):
                model.tracking()

    elif args.model_name in ["detr_pretrained"]:
        from pose_tracking.models.detr import DETRPretrained

        assert args.mt_n_layers == 6, args.mt_n_layers

        model = DETRPretrained(
            use_pretrained_backbone=True,
            num_classes=num_classes,
            rot_out_dim=args.rot_out_dim,
            t_out_dim=args.t_out_dim,
            opt_only=args.opt_only,
            n_layers=args.mt_n_layers,
            dropout=args.dropout,
            dropout_heads=args.dropout_heads,
            head_num_layers=args.rt_mlps_num_layers,
        )
    elif args.model_name in ["detr_basic", "detr_kpt"]:
        detr_args = dict(
            use_pose_tokens=args.mt_use_pose_tokens,
            use_roi=args.mt_use_roi,
            use_depth=args.mt_use_depth,
            do_refinement=args.mt_do_refinement,
            do_refinement_with_pose_token=args.mt_do_refinement_with_pose_token,
            do_refinement_with_attn=args.mt_do_refinement_with_attn,
            num_classes=num_classes,
            n_queries=args.mt_num_queries,
            d_model=args.mt_d_model,
            n_tokens=args.mt_n_tokens,
            n_layers=args.mt_n_layers,
            n_heads=args.mt_n_heads,
            encoding_type=args.mt_encoding_type,
            opt_only=args.opt_only,
            dropout=args.dropout,
            dropout_heads=args.dropout_heads,
            rot_out_dim=args.rot_out_dim,
            t_out_dim=args.t_out_dim,
            head_num_layers=args.rt_mlps_num_layers,
            head_hidden_dim=args.rt_hidden_dim or 256,
            factors=args.factors,
        )
        args.detr_args = argparse.Namespace(**detr_args)

        if args.model_name == "detr_basic":
            from pose_tracking.models.detr import DETR

            model = DETR(
                backbone_name=args.encoder_name,
                **detr_args,
            )
        else:
            from pose_tracking.models.detr import KeypointDETR

            model = KeypointDETR(
                kpt_spatial_dim=args.mt_kpt_spatial_dim,
                use_mask_on_input=args.mt_use_mask_on_input,
                use_mask_as_obj_indicator=False,
                do_calibrate_kpt=args.mt_do_calibrate_kpt,
                **detr_args,
            )
    elif args.model_name == "cvae":
        model = CVAE(
            z_dim=args.cvae_z_dim,
            hidden_dim=args.hidden_dim,
            rt_mlps_num_layers=args.rt_mlps_num_layers,
            encoder_out_dim=args.encoder_out_dim,
            dropout=args.dropout,
            dropout_heads=args.dropout_heads,
            encoder_name=args.encoder_name,
            encoder_img_weights=args.encoder_img_weights,
            norm_layer_type=args.norm_layer_type,
            do_predict_2d_t=args.do_predict_2d_t,
            do_predict_6d_rot=args.do_predict_6d_rot,
            do_predict_3d_rot=args.do_predict_3d_rot,
            use_prev_pose_condition=args.use_prev_pose_condition,
            use_prev_latent=args.use_prev_latent,
            do_predict_t=not args.opt_only or "t" in args.opt_only,
            do_predict_rot=not args.opt_only or "rot" in args.opt_only,
            rt_hidden_dim=args.rt_hidden_dim,
            use_mlp_for_prev_pose=args.use_mlp_for_prev_pose,
            use_depth=args.use_depth,
            num_samples=args.cvae_num_samples,
        )
    else:
        num_pts = 256
        priv_dim = num_pts * 3
        latent_dim = args.encoder_out_dim  # defined by the encoders
        depth_dim = latent_dim
        rgb_dim = latent_dim
        extra_kwargs = {}
        if args.model_name == "cnnlstm":
            model_cls = RecurrentCNN
        elif args.model_name == "cnnlstm_vanilla":
            model_cls = RecurrentCNNVanilla
        elif args.model_name == "cnnlstm_sep":
            model_cls = RecurrentCNNSeparated
        elif args.model_name == "cnn":
            model_cls = CNN
        elif args.model_name == "cnn_kpt":
            model_cls = KeypointCNN
        elif args.model_name == "kpt_pose":
            model_cls = KeypointPose
        elif args.model_name == "kpt_kpt":
            model_cls = KeypointKeypoint
        elif args.model_name == "cnnlstm_double":
            model_cls = RecurrentCNNDouble
            extra_kwargs["use_crop_for_rot"] = args.use_crop_for_rot
        else:
            raise ValueError(f"Unknown model name {args.model_name}")
        if args.model_name != "cnnlstm_vanilla":
            extra_kwargs.update(
                use_obs_belief=args.use_obs_belief,
                use_priv_decoder=args.use_priv_decoder,
                use_belief_decoder=args.use_belief_decoder,
                bdec_priv_decoder_out_dim=priv_dim,
                belief_hidden_dim=args.belief_hidden_dim,
                belief_num_layers=args.belief_num_layers,
            )
        model = model_cls(
            do_predict_2d_t=args.do_predict_2d_t,
            do_predict_6d_rot=args.do_predict_6d_rot,
            do_predict_3d_rot=args.do_predict_3d_rot,
            use_rnn=args.use_rnn,
            do_freeze_encoders=args.do_freeze_encoders,
            use_prev_pose_condition=args.use_prev_pose_condition,
            use_prev_latent=args.use_prev_latent,
            do_predict_kpts=args.do_predict_kpts,
            use_mlp_for_prev_pose=args.use_mlp_for_prev_pose,
            use_kpts_for_rot=args.use_kpts_for_rot,
            do_predict_abs_pose=args.do_predict_abs_pose,
            use_depth=args.use_depth,
            depth_dim=depth_dim,
            rgb_dim=rgb_dim,
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            rnn_type=args.rnn_type,
            rnn_state_init_type=args.rnn_state_init_type,
            encoder_name=args.encoder_name,
            rt_mlps_num_layers=args.rt_mlps_num_layers,
            dropout=args.dropout,
            dropout_heads=args.dropout_heads,
            encoder_depth_weights=args.encoder_depth_weights,
            encoder_img_weights=args.encoder_img_weights,
            norm_layer_type=args.norm_layer_type,
            encoder_out_dim=args.encoder_out_dim,
            r_num_layers_inc=args.r_num_layers_inc,
            rt_hidden_dim=args.rt_hidden_dim,
            bbox_num_kpts=args.bbox_num_kpts,
            do_predict_t=not args.opt_only or "t" in args.opt_only,
            do_predict_rot=not args.opt_only or "rot" in args.opt_only,
            **extra_kwargs,
        )

    if args.ckpt_path:
        print(f"Loading model from {args.ckpt_path}")
        model = load_from_ckpt(args.ckpt_path, model)["model"]
    if args.ckpt_exp_name:
        model_artifact_name = "model_best"
        print(f"Loading model from {model_artifact_name} artifact at {args.ckpt_exp_name}")
        model = load_model_from_exp(model, args.ckpt_exp_name, model_artifact_name=model_artifact_name)

    return model


def get_memotr_args(args):
    config = yaml.unsafe_load(
        open(
            f"{MEMOTR_DIR}/train_mot17.yaml",
            "r",
        )
    )
    config["USE_DAB"] = False
    config["BACKBONE"] = args.encoder_name
    config["rot_out_dim"] = args.rot_out_dim
    config["t_out_dim"] = args.t_out_dim
    config["NUM_DET_QUERIES"] = args.mt_num_queries
    config["NUM_DEC_LAYERS"] = args.mt_n_layers
    config["NUM_ENC_LAYERS"] = args.mt_n_layers
    config["WITH_BOX_REFINE"] = False
    config["WITH_BOX_REFINE"] = True
    config["opt_only"] = args.opt_only

    config["LR_BACKBONE"] = args.lr_encoders
    config["LR_POINTS"] = args.lr * 1e-1
    config["LR"] = args.lr
    return config


def get_trackformer_args(args):
    tf_args = argparse.Namespace(
        **yaml.load(
            open(
                f"{TF_DIR}/config.yaml",
                "r",
            ),
            Loader=yaml.Loader,
        )
    )
    tf_args.deformable = args.tf_use_deformable

    tf_args.focal_loss = args.tf_use_focal_loss
    tf_args.use_kpts = args.tf_use_kpts
    tf_args.use_depth = args.mt_use_depth
    tf_args.use_kpts_as_ref_pt = args.tf_use_kpts_as_ref_pt
    tf_args.use_kpts_as_img = args.tf_use_kpts_as_img

    if args.tf_use_kpts:
        tf_args.lr_backbone_names = ["extractor"]

    tf_args.multi_frame_attention = True
    tf_args.merge_frame_features = args.tf_do_merge_frame_features
    tf_args.overflow_boxes = True
    tf_args.multi_frame_encoding = args.tf_use_multi_frame_encoding

    tf_args.track_query_false_negative_prob = args.tf_track_query_false_negative_prob
    tf_args.track_query_false_positive_prob = args.tf_track_query_false_positive_prob

    tf_args.lr = args.lr
    tf_args.lr_backbone = args.lr_encoders
    tf_args.lr_linear_proj_mult = tf_args.lr_linear_proj_mult
    tf_args.lr_track = tf_args.lr
    tf_args.with_box_refine = args.tf_use_box_refine
    tf_args.num_queries = args.mt_num_queries
    tf_args.enc_layers = args.mt_n_layers
    tf_args.dec_layers = args.mt_n_layers
    tf_args.dropout = args.dropout
    tf_args.dropout_heads = args.dropout_heads
    tf_args.hidden_dim = args.tf_transformer_hidden_dim
    tf_args.rot_out_dim = args.rot_out_dim
    tf_args.t_out_dim = args.t_out_dim

    tf_args.bbox_loss_coef = args.tf_bbox_loss_coef
    tf_args.giou_loss_coef = args.tf_giou_loss_coef
    tf_args.ce_loss_coef = args.tf_ce_loss_coef
    tf_args.rot_loss_coef = args.tf_rot_loss_coef
    tf_args.t_loss_coef = args.tf_t_loss_coef
    tf_args.depth_loss_coef = args.tf_depth_loss_coef

    tf_args.opt_only = args.opt_only
    tf_args.factors = args.factors

    tf_args.backbone = args.encoder_name
    tf_args.head_num_layers = args.rt_mlps_num_layers

    if args.tf_use_kpts_as_img:
        tf_args.num_feature_levels = 2

    return tf_args


def get_trainer(
    args, model, device="cuda", writer=None, world_size=1, logger=None, do_vis=False, exp_dir=None, num_classes=None
):
    from pose_tracking.trainer import Trainer
    from pose_tracking.trainer_cvae import TrainerCVAE
    from pose_tracking.trainer_detr import TrainerDeformableDETR, TrainerTrackformer
    from pose_tracking.trainer_kpt import TrainerKeypoints
    from pose_tracking.trainer_others import TrainerPizza, TrainerVideopose

    criterion_trans = get_t_loss(args.t_loss_name)
    criterion_rot = get_rot_loss(args.rot_loss_name)
    use_pose_loss = args.pose_loss_name in ["add"]
    criterion_pose = compute_add_loss if use_pose_loss else None

    if args.model_name in ["videopose"]:
        trainer_cls = TrainerVideopose
    elif args.model_name in ["pizza"]:
        assert args.do_predict_rel_pose and args.t_repr == "2d" and args.rot_repr == "axis_angle"
        trainer_cls = TrainerPizza
        # trainer_cls = Trainer
    elif "detr" in args.model_name:
        trainer_cls = TrainerDeformableDETR
    elif "cvae" in args.model_name:
        trainer_cls = TrainerCVAE
    elif any(x in args.model_name for x in ["cnn_kpt", "kpt_kpt"]):
        trainer_cls = TrainerKeypoints
    elif "memotr" in args.model_name:
        from pose_tracking.trainer_memotr import TrainerMemotr

        trainer_cls = TrainerMemotr
    elif "trackformer" in args.model_name:
        trainer_cls = TrainerTrackformer
    else:
        trainer_cls = Trainer

    is_detr = "detr" in args.model_name or args.model_name in ["trackformer", "memotr"]
    if is_detr:
        assert num_classes is not None
        extra_kwargs = {
            "num_classes": num_classes,
            "num_dec_layers": args.mt_n_layers,
            "use_pose_tokens": args.mt_use_pose_tokens,
            "aux_loss": True,
        }
        if "detr_kpt" in args.model_name:
            extra_kwargs.update(
                {
                    "kpt_spatial_dim": args.mt_kpt_spatial_dim,
                    "do_calibrate_kpt": args.mt_do_calibrate_kpt,
                }
            )
        if "memotr" in args.model_name:
            extra_kwargs.update({"config": args.memotr_args})
        extra_kwargs.update({"args": args})
    elif args.model_name in ["cvae"]:
        extra_kwargs = {"kl_loss_coef": args.cvae_kl_loss_coef}
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
        use_rnn=args.use_rnn,
        use_belief_decoder=args.use_depth and args.use_obs_belief and args.use_belief_decoder,
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
        max_clip_grad_norm=args.max_clip_grad_norm,
        do_chunkify_val=args.do_chunkify_val,
        do_perturb_init_gt_for_rel_pose=args.do_perturb_init_gt_for_rel_pose,
        tf_t_loss_coef=args.tf_t_loss_coef,
        tf_rot_loss_coef=args.tf_rot_loss_coef,
        include_abs_pose_loss_for_rel=args.include_abs_pose_loss_for_rel,
        use_entire_seq_in_train=args.use_entire_seq_in_train,
        use_seq_len_curriculum=args.use_seq_len_curriculum,
        do_predict_abs_pose=args.do_predict_abs_pose,
        use_pnp_for_rot_pred=args.use_pnp_for_rot_pred,
        bbox_num_kpts=args.bbox_num_kpts,
        use_factors=args.use_factors,
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
    model_name,
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
    ds_types=("train", "val", "train_as_val"),
    do_predict_kpts=False,
    use_priv_decoder=False,
    do_overfit=False,
    do_preload_ds=False,
    do_subtract_bg=False,
    include_mask=False,
    include_depth=False,
    include_bbox_2d=False,
    do_predict_rel_pose=False,
    use_entire_seq_in_train=False,
    do_normalize_depth=False,
    use_mask_for_bbox_2d=True,
    max_train_videos=None,
    max_val_videos=None,
    max_test_videos=None,
    start_frame_idx=0,
    end_frame_idx=None,
    rot_repr="quaternion",
    t_repr="3d",
    max_random_seq_step=4,
    seq_len_max_train=100,
    seq_len_max_val=200,
    max_depth_m=10,
    dino_features_folder_name=None,
    bbox_num_kpts=32,
    do_load_mesh_in_memory=False,
    factors=None,
):

    transform_rgb = get_transforms(transform_names, transform_prob=transform_prob) if transform_names else None
    is_tf_model = "trackformer" in model_name
    is_cnnlstm_model = "cnnlstm" in model_name
    is_detr_model = "detr" in model_name or is_tf_model
    is_detr_kpt_model = "detr_kpt" in model_name
    is_roi_model = "_sep" in model_name
    is_pizza_model = "pizza" in model_name
    is_motr_model = "motr" in model_name
    include_bbox_2d = (
        do_predict_kpts or is_detr_model or is_roi_model or is_pizza_model or include_bbox_2d or is_motr_model
    )
    include_mask = include_mask or do_subtract_bg
    ds_kwargs_common = dict(
        shorter_side=None,
        zfar=max_depth_m,
        include_mask=include_mask,
        include_depth=include_depth or is_cnnlstm_model or is_detr_kpt_model,
        include_bbox_2d=include_bbox_2d,
        start_frame_idx=start_frame_idx,
        mask_pixels_prob=mask_pixels_prob,
        do_normalize_bbox=True if is_detr_model or is_tf_model or is_motr_model else False,
        bbox_format="cxcywh" if is_detr_model or is_motr_model else "xyxy",
        model_name="pizza" if is_pizza_model else model_name,
        do_normalize_depth=True if (is_cnnlstm_model or do_normalize_depth) else False,
        rot_repr=rot_repr,
        t_repr=t_repr,
        do_subtract_bg=do_subtract_bg,
        bbox_num_kpts=bbox_num_kpts,
        dino_features_folder_name=dino_features_folder_name,
        use_mask_for_bbox_2d=use_mask_for_bbox_2d,
        use_occlusion_augm="occlusion" in transform_names if transform_names else False,
        do_load_mesh_in_memory=do_load_mesh_in_memory,
        factors=factors,
    )
    if ds_name in ["ycbi"]:
        ycbi_kwargs = dict(
            ycb_meshes_dir=YCB_MESHES_DIR,
        )
        ds_kwargs_custom = ycbi_kwargs
    elif ds_name in ["ikea", "custom", "custom_test", "ho3d_v3", "ikea_multiobj", "ycbv"]:
        ds_kwargs_custom = dict()
    else:
        cube_sim_kwargs = dict(
            mesh_path=f"{ds_video_dir_train}/mesh/cube.obj",
        )
        ds_kwargs_custom = cube_sim_kwargs

    ds_kwargs = {**ds_kwargs_common, **ds_kwargs_custom}

    res = {}

    video_ds_cls = VideoDatasetTracking if (is_tf_model or is_detr_model or is_motr_model) else VideoDataset

    train_ds_kwargs = copy.deepcopy(ds_kwargs)
    if "train" in ds_types:
        train_ds_kwargs["video_dir"] = Path(ds_video_dir_train)
        train_ds_kwargs["is_val"] = False
        train_dataset = get_video_ds(
            ds_video_subdirs=ds_video_subdirs_train,
            ds_name=ds_name,
            seq_len=seq_len_max_train if use_entire_seq_in_train else seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            ds_kwargs=train_ds_kwargs,
            num_samples=num_samples,
            do_preload=do_preload_ds,
            transforms_rgb=transform_rgb,
            video_ds_cls=video_ds_cls,
            max_videos=max_train_videos,
            max_random_seq_step=max_random_seq_step,
            do_predict_rel_pose=do_predict_rel_pose,
            end_frame_idx=end_frame_idx,
        )
        mesh_paths_orig_stems_train = [Path(d.ds.mesh_path_orig).stem for d in train_dataset.video_datasets]
        res["train"] = train_dataset
    else:
        mesh_paths_orig_stems_train = None

    if "val" in ds_types:
        if do_overfit:
            val_ds_kwargs_full = dict(
                ds_video_subdirs=ds_video_subdirs_train,
                ds_name=ds_name,
                seq_len=seq_len_max_val,
                seq_step=1,
                seq_start=0,
                ds_kwargs=train_ds_kwargs,
                num_samples=None,
                do_preload=do_preload_ds,
                transforms_rgb=None,
                video_ds_cls=video_ds_cls,
                max_videos=max_val_videos,
                max_random_seq_step=max_random_seq_step,
                do_predict_rel_pose=do_predict_rel_pose,
                end_frame_idx=end_frame_idx,
            )
            val_dataset = get_video_ds(**val_ds_kwargs_full)
            if "train_as_val" in ds_types:
                train_as_val_ds_kwargs_full = copy.deepcopy(val_ds_kwargs_full)
                train_as_val_ds_kwargs_full["max_videos"] = 1
                res["train_as_val"] = get_video_ds(**train_as_val_ds_kwargs_full)
            assert (
                set(vds.ds.video_dir for vds in val_dataset.video_datasets)
                - set(vds.ds.video_dir for vds in train_dataset.video_datasets)
                == set()
            )
        else:
            assert ds_video_dir_val and ds_video_subdirs_val, print(f"{ds_video_dir_val=} {ds_video_subdirs_val}")
            val_ds_kwargs = copy.deepcopy(ds_kwargs)
            val_ds_kwargs.pop("mask_pixels_prob")
            val_ds_kwargs["video_dir"] = Path(ds_video_dir_val)
            val_ds_kwargs_full = dict(
                ds_video_subdirs=ds_video_subdirs_val,
                ds_name=ds_name,
                seq_len=seq_len,
                seq_step=seq_step,
                seq_start=None,
                ds_kwargs=val_ds_kwargs,
                num_samples=num_samples,
                do_preload=do_preload_ds,
                mesh_paths_to_take_stems=mesh_paths_orig_stems_train,
                video_ds_cls=video_ds_cls,
                max_videos=max_val_videos,
                max_random_seq_step=max_random_seq_step,
                do_predict_rel_pose=do_predict_rel_pose,
            )
            val_dataset = get_video_ds(**val_ds_kwargs_full)
            if "train_as_val" in ds_types:
                train_as_val_ds_kwargs_full = copy.deepcopy(val_ds_kwargs_full)
                train_as_val_ds_kwargs_full["ds_kwargs"]["video_dir"] = Path(ds_video_dir_train)
                train_as_val_ds_kwargs_full["ds_video_subdirs"] = ds_video_subdirs_train
                res["train_as_val"] = get_video_ds(**train_as_val_ds_kwargs_full)

        res["val"] = val_dataset

    if "test" in ds_types:
        assert ds_video_dir_test and ds_video_subdirs_test, print(f"{ds_video_dir_test=} {ds_video_subdirs_test}")
        test_ds_kwargs = copy.deepcopy(ds_kwargs)
        test_ds_kwargs.pop("mask_pixels_prob")
        test_ds_kwargs["video_dir"] = Path(ds_video_dir_test)
        test_dataset = get_video_ds(
            ds_video_subdirs=ds_video_subdirs_test,
            ds_name=ds_name,
            seq_len=seq_len,
            seq_step=1,
            seq_start=0,
            ds_kwargs=test_ds_kwargs,
            num_samples=num_samples,
            do_preload=do_preload_ds,
            video_ds_cls=video_ds_cls,
            max_videos=max_test_videos,
            max_random_seq_step=max_random_seq_step,
            do_predict_rel_pose=do_predict_rel_pose,
            end_frame_idx=end_frame_idx,
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
    mesh_paths_to_take_stems=None,
    video_ds_cls=VideoDataset,
    max_videos=None,
    end_frame_idx=None,
    max_random_seq_step=8,
    do_predict_rel_pose=False,
):
    video_datasets = []
    count = 0
    for ds_video_subdir in tqdm(ds_video_subdirs, leave=False, desc="Video datasets"):
        ds_kwargs["end_frame_idx"] = end_frame_idx
        ds = get_obj_ds(ds_name, ds_kwargs, ds_video_subdir=ds_video_subdir)
        seq_len = len(ds) if seq_len is None else seq_len
        if mesh_paths_to_take_stems is not None:
            if Path(ds.mesh_path_orig).stem not in mesh_paths_to_take_stems:
                continue
        video_ds = video_ds_cls(
            ds=ds,
            seq_len=seq_len,
            seq_step=seq_step,
            seq_start=seq_start,
            num_samples=num_samples,
            do_preload=do_preload,
            transforms_rgb=transforms_rgb,
            max_random_seq_step=max_random_seq_step,
            do_predict_rel_pose=do_predict_rel_pose,
        )
        video_datasets.append(video_ds)
        count += 1
        if max_videos is not None and count >= max_videos:
            break

    full_ds = MultiVideoDataset(video_datasets)
    return full_ds


def get_obj_ds(ds_name, ds_kwargs, ds_video_subdir):
    ds_kwargs = ds_kwargs.copy()
    ds_kwargs["video_dir"] = Path(ds_kwargs["video_dir"]) / ds_video_subdir
    model_name = ds_kwargs.pop("model_name", None)
    if ds_name == "ycbi":
        if model_name == "pizza":
            ds_cls = YCBineoatDataset
        else:
            ds_cls = YCBineoatDataset
        ds = ds_cls(**ds_kwargs)
    elif ds_name == "cube_sim":
        ds = CustomSimDatasetCube(**ds_kwargs)
    elif ds_name == "custom":
        ds = CustomDataset(**ds_kwargs)
    elif ds_name == "custom_test":
        ds = CustomDatasetTest(**ds_kwargs)
    elif ds_name == "ho3d_v3":
        ds = HO3DDataset(**ds_kwargs)
    elif ds_name == "ycbv":
        ds = YCBvDataset(**ds_kwargs)
    elif ds_name == "ikea":
        ds = CustomSimDatasetIkea(**ds_kwargs)
    elif ds_name == "ikea_multiobj":
        ds = CustomSimMultiObjDatasetIkea(**ds_kwargs)
    else:
        raise NotImplementedError(f"Unknown dataset name {ds_name}")
    return ds


def get_ds_dirs(args):
    get_ds_root_dirs_res = get_ds_root_dirs(args)
    ds_video_dir_train = get_ds_root_dirs_res["ds_video_dir_train"]
    ds_video_dir_val = get_ds_root_dirs_res["ds_video_dir_val"]

    if args.ds_name in ["ycbi", "cube"]:
        ds_video_subdirs_train = args.obj_names
        ds_video_subdirs_val = args.obj_names_val
    elif args.ds_name in ["custom"]:
        ds_video_subdirs_train = ["cube_data"]
        ds_video_subdirs_val = ["cube_data"]
    elif args.ds_name in ["custom_test"]:
        ds_video_subdirs_train = args.obj_names
        ds_video_subdirs_val = args.obj_names_val
    elif args.ds_name in ["ho3d_v3", "ycbv"]:
        ds_video_subdirs_train = [Path(p).name for p in get_ordered_paths(ds_video_dir_train / "*")]
        ds_video_subdirs_val = [Path(p).name for p in get_ordered_paths(ds_video_dir_val / "*")]
    else:
        ds_video_subdirs_train = [Path(p).name for p in get_ordered_paths(ds_video_dir_train / "env_*")]
        ds_video_subdirs_val = [Path(p).name for p in get_ordered_paths(ds_video_dir_val / "env_*")]
    # ds_video_subdirs_train = [x for x in ds_video_subdirs_train if x in ['env_443']]
    if "cube_500_random_realsense" in str(ds_video_dir_train):
        ds_video_subdirs_val = [x for x in ds_video_subdirs_val if x not in ["env_16"]]
    if "cube_500_random_v3" in str(ds_video_dir_train):
        ds_video_subdirs_val = [x for x in ds_video_subdirs_val if x not in ["env_5"]]

    if args.do_split_train_for_val:
        assert args.val_split_share > 0
        ds_video_subdirs_train = ds_video_subdirs_train + ds_video_subdirs_val
        val_num_subdirs = int(len(ds_video_subdirs_train) * args.val_split_share)
        val_random_idxs = np.random.choice(len(ds_video_subdirs_train), val_num_subdirs, replace=False)
        ds_video_subdirs_val = [ds_video_subdirs_train[i] for i in val_random_idxs]
        ds_video_subdirs_train = [d for d in ds_video_subdirs_train if d not in ds_video_subdirs_val]
        ds_video_dir_val = ds_video_dir_train

    excluded_envs = json.load(open(PROJ_DIR / "excluded_envs.json", "r"))
    if ds_video_dir_train.name in excluded_envs:
        ds_video_subdirs_train = [d for d in ds_video_subdirs_train if d not in excluded_envs[ds_video_dir_train.name]]
    if ds_video_dir_val.name in excluded_envs:
        ds_video_subdirs_val = [d for d in ds_video_subdirs_val if d not in excluded_envs[ds_video_dir_val.name]]

    return {
        "ds_video_dir_train": ds_video_dir_train,
        "ds_video_dir_val": ds_video_dir_val,
        "ds_video_subdirs_train": ds_video_subdirs_train,
        "ds_video_subdirs_val": ds_video_subdirs_val,
    }


def get_ds_root_dirs(args):
    if args.ds_name in ["ikea", "cube", "custom", "custom_test", "ikea_multiobj"]:
        ds_video_dir_train = DATA_DIR / args.ds_folder_name_train
        ds_video_dir_val = DATA_DIR / args.ds_folder_name_val
    elif args.ds_name in ["ho3d_v3"]:
        ds_video_dir_train = HO3D_ROOT / args.ds_folder_name_train
        ds_video_dir_val = HO3D_ROOT / args.ds_folder_name_val
    elif args.ds_name == "ycbi":
        ds_video_dir_train = YCBINEOAT_SCENE_DIR
        ds_video_dir_val = YCBINEOAT_SCENE_DIR
    elif args.ds_name == "ycbv":
        ds_video_dir_train = YCBV_SCENE_DIR / "train_real"
        ds_video_dir_val = YCBV_SCENE_DIR / "test"
    elif args.ds_name == "nocs":
        ds_video_dir_train = NOCS_SCENE_DIR
        ds_video_dir_val = NOCS_SCENE_DIR
    else:
        raise NotImplementedError(f"Unknown dataset name {args.ds_name}")
    return {
        "ds_video_dir_train": ds_video_dir_train,
        "ds_video_dir_val": ds_video_dir_val,
    }


class Printer:
    def __init__(self, logger=None):
        self.log_fn = logger.info if logger else print

    def saved_artifacts(self, epoch):
        self.log_fn(f"Saved artifacts on epoch {epoch}")

    def print_stats(self, train_stats, stage):
        self.log_fn(f"## {stage.upper()} ##")
        LOSS_METRICS = [k for k in train_stats if "loss" in k]
        ERROR_METRICS = [k for k in train_stats if "err" in k]
        ADDITIONAL_METRICS = [
            "add",
            "adds",
            "miou",
            "5deg5cm",
            "2deg2cm",
            "add_abs",
            "5deg5cm_abs",
            "t_log_likelihood",
            "rot_log_likelihood",
        ]

        for stat_group in [LOSS_METRICS, ERROR_METRICS, ADDITIONAL_METRICS]:
            msg = []
            for k in stat_group:
                if k in train_stats:
                    msg.append(f"{k}: {train_stats[k]:.4f}")
            self.log_fn(" | ".join(msg))
