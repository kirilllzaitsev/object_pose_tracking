import argparse
import base64
import re
import sys

import numpy as np
import yaml


def parse_args():
    parser = get_parser()

    args_raw, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"WARNING. Unknown arguments: {unknown_args}")
    args = postprocess_args(args_raw)
    args_to_group_map = map_args_to_groups(parser, args)
    return args, args_to_group_map


def get_parser():
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("Training arguments")

    pipe_args.add_argument("--exp_disabled", action="store_true", help="Disable experiment logging.")
    pipe_args.add_argument("--do_overfit", action="store_true", help="Overfit setting")
    pipe_args.add_argument("--do_debug", action="store_true", help="Debugging setting")
    pipe_args.add_argument("--use_test_set", action="store_true", help="Predict on a test set")
    pipe_args.add_argument("--do_profile", action="store_true", help="Profile the code")
    pipe_args.add_argument("--do_save_artifacts", action="store_true", help="Save artifacts")
    pipe_args.add_argument("--do_print_seq_stats", action="store_true", help="Print sequence-level stats")
    pipe_args.add_argument(
        "--do_ignore_file_args_with_provided", action="store_true", help="Ignore file args if provided via CLI"
    )
    pipe_args.add_argument("--exp_tags", nargs="*", default=[], help="Tags for the experiment to log.")
    pipe_args.add_argument("--exp_name", type=str, default="test", help="Name of the experiment.")
    pipe_args.add_argument("--device", type=str, default="cuda", help="Device to use")
    pipe_args.add_argument("--args_path", type=str, help="Path to a yaml file with arguments")
    pipe_args.add_argument("--ckpt_path", type=str, help="Path to model checkpoint to load")
    pipe_args.add_argument("--ckpt_exp_name", type=str, help="Name of the experiment to load ckpt from")
    pipe_args.add_argument("--args_from_exp_name", type=str, help="Exp name to load args from")
    pipe_args.add_argument(
        "--ignored_file_args",
        nargs="*",
        default=[],
        help="List of args to ignore when loading from file.",
    )
    pipe_args.add_argument(
        "--provided_ignored_args",
        nargs="*",
        default=[],
        help="Filled in by the code.",
    )

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--use_ddp", action="store_true", help="Use Distributed Data Parallel")
    train_args.add_argument(
        "--is_ddp_interactive", action="store_true", help="SLURM is running in the interactive mode"
    )
    train_args.add_argument("--use_es", action="store_true", help="Use early stopping")
    train_args.add_argument("--do_log_every_ts", action="store_true", help="Log every timestep")
    train_args.add_argument("--do_log_every_seq", action="store_true", help="Log every sequence")
    train_args.add_argument("--do_vis", action="store_true", help="Visualize inputs")
    train_args.add_argument("--use_lrs", action="store_true", help="Use learning rate scheduler")
    train_args.add_argument(
        "--do_perturb_init_gt_for_rel_pose",
        action="store_true",
        help="Apply noise to init gt pose for relative pose estimation",
    )
    train_args.add_argument(
        "--include_abs_pose_loss_for_rel",
        action="store_true",
        help="Include additional abs pose loss when training with relative pose",
    )
    train_args.add_argument(
        "--do_chunkify_val", action="store_true", help="Chunkify validation into train's seq_length"
    )
    train_args.add_argument("--vis_epoch_freq", type=int, default=3, help="Visualize a random sequence every N epochs")
    train_args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    train_args.add_argument("--val_epoch_freq", type=int, default=1, help="Validate every N epochs")
    train_args.add_argument("--save_epoch_freq", type=int, default=1, help="Save model every N epochs")
    train_args.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    train_args.add_argument("--seed", type=int, default=10, help="Random seed")
    train_args.add_argument("--es_patience_epochs", type=int, default=3, help="Early stopping patience")
    train_args.add_argument("--es_delta", type=float, default=1e-4, help="Early stopping delta")
    train_args.add_argument("--lr", type=float, default=1e-4, help="lr for the rest of the model")
    train_args.add_argument("--lr_encoders", type=float, default=1e-4, help="lr for encoders")
    train_args.add_argument(
        "--lrs_type", type=str, default="pl", help="Type of lr scheduler", choices=["pl", "step", "none"]
    )
    train_args.add_argument("--lrs_step_size", type=int, default=10, help="Number of epochs before changing lr")
    train_args.add_argument("--lrs_gamma", type=float, default=0.5, help="Scaler for lr scheduler")
    train_args.add_argument("--lrs_min_lr", type=float, default=5e-6, help="Minimum lr")
    train_args.add_argument("--lrs_patience", type=int, default=3, help="Patience for lr scheduler")
    train_args.add_argument("--lrs_delta", type=float, default=0.0, help="Delta between scores for lr scheduler")
    train_args.add_argument("--lrs_threshold_mode", type=str, default="abs", help="Threshold mode for lr scheduler")
    train_args.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    train_args.add_argument("--max_clip_grad_norm", type=float, default=0.1, help="Max grad norm")
    train_args.add_argument(
        "--pose_loss_name", type=str, default="separate", help="Pose loss name", choices=["separate", "add"]
    )
    train_args.add_argument(
        "--t_loss_name",
        type=str,
        default="mse",
        help="Translation loss name",
        choices=["mse", "rmse", "mae", "huber", "huber_norm", "angle", "mixed"],
    )
    train_args.add_argument(
        "--rot_loss_name",
        type=str,
        default="mse",
        help="Rotation loss name",
        choices=["geodesic", "mse", "rmse", "mae", "huber", "videopose", "displacement", "geodesic_mat"],
    )
    train_args.add_argument(
        "--opt_only",
        nargs="*",
        help="List of tasks to optimize",
        default=["rot", "t", "labels", "boxes", "depth"],
        choices=["rot", "t", "labels", "boxes", "depth"],
    )

    poseformer_args = parser.add_argument_group("PoseFormer arguments")
    poseformer_args.add_argument("--mt_do_calibrate_kpt", action="store_true", help="Calibrate keypoints")
    poseformer_args.add_argument("--mt_use_pose_tokens", action="store_true", help="Use pose tokens")
    poseformer_args.add_argument(
        "--mt_use_mask_on_input", action="store_true", help="Mask out non-object part of the image (dilated)"
    )
    poseformer_args.add_argument(
        "--mt_kpt_spatial_dim",
        type=int,
        default=2,
        help="Spatial dimension of keypoints",
        choices=[2, 3],
    )
    poseformer_args.add_argument(
        "--mt_encoding_type",
        type=str,
        default="spatial",
        help="Encoding type for positional encoding",
        choices=["spatial", "sin", "learned", "none"],
    )
    poseformer_args.add_argument(
        "--mt_num_queries",
        type=int,
        default=100,
        help="Number of object queries for the transformer",
    )
    poseformer_args.add_argument(
        "--mt_d_model",
        type=int,
        default=256,
        help="Number of features for the transformer",
    )
    poseformer_args.add_argument(
        "--mt_n_tokens",
        type=int,
        default=300,
        help="Number of tokens for the transformer",
    )
    poseformer_args.add_argument(
        "--mt_n_layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    poseformer_args.add_argument(
        "--mt_n_heads",
        type=int,
        default=8,
        help="Number of transformer heads",
    )

    tf_args = parser.add_argument_group("Trackformer arguments")
    tf_args.add_argument("--tf_use_deformable", action="store_true", help="Use deformable detr")
    tf_args.add_argument("--tf_use_multi_frame_encoding", action="store_true", help="Use multi_frame_encoding")
    tf_args.add_argument("--tf_do_merge_frame_features", action="store_true", help="Concat frame features at t-1 and t")
    tf_args.add_argument("--tf_use_box_refine", action="store_true", help="Use box refinement")
    tf_args.add_argument("--tf_use_focal_loss", action="store_true", help="Use focal_loss")
    tf_args.add_argument("--tf_use_kpts", action="store_true", help="Use keypoints in tf")
    tf_args.add_argument(
        "--tf_use_kpts_as_ref_pt", action="store_true", help="Use keypoints as reference points for encoder"
    )
    tf_args.add_argument("--tf_use_kpts_as_img", action="store_true", help="Use keypoints instead of images")
    tf_args.add_argument("--tf_bbox_loss_coef", type=float, default=5)
    tf_args.add_argument("--tf_giou_loss_coef", type=float, default=2)
    tf_args.add_argument("--tf_ce_loss_coef", type=float, default=2)
    tf_args.add_argument("--tf_rot_loss_coef", type=float, default=1)
    tf_args.add_argument("--tf_t_loss_coef", type=float, default=1)
    tf_args.add_argument("--tf_track_query_false_negative_prob", type=float, default=0.4)
    tf_args.add_argument("--tf_track_query_false_positive_prob", type=float, default=0.1)
    tf_args.add_argument("--tf_depth_loss_coef", type=float, default=1)

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument(
        "--do_predict_2d_t", action="store_true", help="Predict object 2D center and depth separately"
    )
    model_args.add_argument("--do_predict_6d_rot", action="store_true", help="Predict object rotation as 6D")
    model_args.add_argument("--do_predict_3d_rot", action="store_true", help="Predict object rotation as 3D")
    model_args.add_argument("--do_predict_rel_pose", action="store_true", help="Predict relative pose")
    model_args.add_argument("--do_predict_abs_pose", action="store_true", help="Predict absolute pose in addition to relative pose")
    model_args.add_argument("--do_predict_kpts", action="store_true", help="Predict keypoints")
    model_args.add_argument("--use_prev_latent", action="store_true", help="Use t-1 latent as condition")
    model_args.add_argument("--no_rnn", action="store_true", help="Use a simple MLP instead of RNN")
    model_args.add_argument("--use_priv_decoder", action="store_true", help="Use privileged info decoder")
    model_args.add_argument("--use_mlp_for_prev_pose", action="store_true", help="Project previous rot/t with MLP")
    model_args.add_argument("--do_freeze_encoders", action="store_true", help="Whether to freeze encoder backbones")
    model_args.add_argument("--use_prev_pose_condition", action="store_true", help="Use previous pose as condition")
    model_args.add_argument(
        "--use_pretrained_model", action="store_true", help="Use a pretrained model of the same architecture"
    )
    model_args.add_argument(
        "--no_obs_belief", action="store_true", help="Do not use observation belief encoder-decoder"
    )
    model_args.add_argument("--use_belief_decoder", action="store_true")
    model_args.add_argument(
        "--model_name",
        type=str,
        default="cnnlstm",
        help="Model name",
        choices=[
            "cnnlstm",
            "cnnlstm_sep",
            "cnnlstm_vanilla",
            "videopose",
            "detr",
            "detr_basic",
            "detr_kpt",
            "pizza",
            "detr_pretrained",
            "trackformer",
        ],
    )
    model_args.add_argument("--rnn_type", type=str, default="gru", help="RNN type", choices=["gru", "lstm"])
    model_args.add_argument(
        "--rnn_state_init_type",
        type=str,
        default="zeros",
        help="RNN state initialization type",
        choices=["zeros", "rand", "learned"],
    )
    model_args.add_argument("--encoder_name", type=str, default="resnet18", help="Encoder name for both RGB and depth")
    model_args.add_argument("--encoder_img_weights", type=str, help="Weights for the image encoder")
    model_args.add_argument("--encoder_depth_weights", type=str, help="Weights for the depth encoder")
    model_args.add_argument(
        "--norm_layer_type",
        type=str,
        default="frozen_bn",
        help="Type of normalization layer for encoders",
        choices=["frozen_bn", "bn", "id"],
    )
    model_args.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension across the model")
    model_args.add_argument(
        "--rt_hidden_dim", type=int, help="Hidden dimension for rot/translation MLPs. Defaults to hidden_dim"
    )
    model_args.add_argument(
        "--benc_belief_enc_hidden_dim", type=int, default=256, help="Hidden dimension for belief encoder"
    )
    model_args.add_argument(
        "--benc_belief_depth_enc_hidden_dim", type=int, default=256, help="Hidden dimension for belief depth encoder"
    )
    model_args.add_argument(
        "--benc_belief_enc_num_layers", type=int, default=2, help="Number of layers for belief encoder"
    )
    model_args.add_argument(
        "--benc_belief_depth_enc_num_layers", type=int, default=2, help="Number of layers for belief depth encoder"
    )
    model_args.add_argument(
        "--bdec_priv_decoder_hidden_dim", type=int, default=256, help="Hidden dimension for privileged info decoder"
    )
    model_args.add_argument(
        "--bdec_depth_decoder_hidden_dim", type=int, default=256, help="Hidden dimension for depth decoder"
    )
    model_args.add_argument(
        "--bdec_hidden_attn_hidden_dim", type=int, default=256, help="Hidden dimension for hidden attention"
    )
    model_args.add_argument(
        "--encoder_out_dim", type=int, default=512, help="Output dimension of the img/depth encoder"
    )
    model_args.add_argument(
        "--priv_decoder_num_layers", type=int, default=2, help="Number of layers for privileged info decoder"
    )
    model_args.add_argument(
        "--depth_decoder_num_layers", type=int, default=2, help="Number of layers for depth decoder from hidden state"
    )
    model_args.add_argument(
        "--hidden_attn_num_layers", type=int, default=2, help="Number of layers for hidden attention in decoder"
    )
    model_args.add_argument(
        "--rt_mlps_num_layers", type=int, default=2, help="Number of layers for rotation and translation MLPs"
    )
    model_args.add_argument(
        "--r_num_layers_inc", type=int, default=0, help="Number of layers to add to the rotation MLP"
    )
    model_args.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the model")
    model_args.add_argument("--dropout_heads", type=float, default=0.0, help="Dropout rate for rot/t/bbox/clf MLPs")

    data_args = parser.add_argument_group("Data arguments")
    data_args.add_argument("--do_preload_ds", action="store_true", help="Preload videos")
    data_args.add_argument("--do_subtract_bg", action="store_true", help="Subtract background from RGBD")
    data_args.add_argument(
        "--use_entire_seq_in_train",
        action="store_true",
        help="Instead of seq_len, use seq_len_max_val for train sequences",
    )
    data_args.add_argument(
        "--do_split_train_for_val", action="store_true", help="Obtain train/val by splitting the train set"
    )
    data_args.add_argument(
        "--use_seq_len_curriculum", action="store_true", help="Gradually increase seq_len during training"
    )
    data_args.add_argument(
        "--val_split_share",
        type=float,
        default=0.1,
        help="Share of the train set to use for validation. Applies only if do_split_train_for_val is set",
    )
    data_args.add_argument("--seq_len", type=int, default=5, help="Number of frames to take for train/val")
    data_args.add_argument("--seq_len_test", type=int, default=600, help="Number of frames to take for test")
    data_args.add_argument("--seq_start", type=int, help="Start frame index in a sequence")
    data_args.add_argument("--seq_step", type=int, default=1, help="Step between frames in a sequence")
    data_args.add_argument("--max_train_videos", type=int, default=1000, help="Max number of videos for training")
    data_args.add_argument("--max_val_videos", type=int, default=100, help="Max number of videos for validation")
    data_args.add_argument(
        "--num_samples",
        type=int,
        help="Number of times to fetch sequences (len(video_ds)). Defaults to len(ds)//seq_step",
    )
    data_args.add_argument("--max_random_seq_step", type=int, default=4, help="Max random step when sampling sequences")
    data_args.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    data_args.add_argument("--num_classes", type=int, help="Hard-coded number of classes")
    data_args.add_argument("--seq_len_max_train", type=int, default=100, help="Number of timesteps in the entire train seq (splitted into seq_len if use_entire_seq_in_train)")
    data_args.add_argument("--obj_names", nargs="*", default=[], help="Object names to use in the dataset")
    data_args.add_argument(
        "--obj_names_val",
        nargs="*",
        default=[],
        help="Object names to use in the validation dataset",
    )
    data_args.add_argument(
        "--ds_name", type=str, default="ycbi", help="Dataset name", choices=["ycbi", "cube_sim", "ikea", "ho3d_v3"]
    )
    data_args.add_argument(
        "--end_frame_idx", type=int, help="Optional index of the last frame of each tracking ds for train set"
    )
    data_args.add_argument("--ds_alias", type=str, help="Optional alias for ds")
    data_args.add_argument("--ds_folder_name_train", type=str, help="Name of the folder with the train dataset")
    data_args.add_argument("--ds_folder_name_val", type=str, help="Name of the folder with the val dataset")
    data_args.add_argument(
        "--mask_pixels_prob", type=float, default=0.0, help="Probability of masking pixels in RGB/depth"
    )
    data_args.add_argument(
        "--transform_names",
        nargs="*",
        default=[],
        help="List of transform names to use",
        choices=["jitter", "iso", "brightness", "blur", "motion_blur", "gamma", "hue", "norm"],
    )
    data_args.add_argument("--transform_prob", type=float, default=0.0, help="Probability of applying the transforms")
    return parser


def postprocess_args(args, use_if_provided=True):

    args = fix_outdated_args(args)

    if use_if_provided and (args.args_path or getattr(args, "args_from_exp_name")):
        if getattr(args, "args_from_exp_name"):
            from pose_tracking.utils.comet_utils import load_artifacts_from_comet

            print(f"Overriding with args from exp {args.args_from_exp_name}")
            loaded_args = load_artifacts_from_comet(args.args_from_exp_name, do_load_model=False)["args"]
            loaded_args = vars(loaded_args)
        else:
            import yaml

            with open(args.args_path, "r") as f:
                loaded_args = yaml.load(f, Loader=yaml.FullLoader)

        default_ignored_file_args = [
            "device",
            "exp_name",
            "use_ddp",
            "exp_disabled",
            "batch_size",
        ]
        if args.do_ignore_file_args_with_provided:
            provided_ignored_args = re.findall("--(\w+)", " ".join(sys.argv[1:]))
        else:
            provided_ignored_args = []
        args.provided_ignored_args += provided_ignored_args
        ignored_file_args = (
            set(args.provided_ignored_args) | set(default_ignored_file_args) | set(args.ignored_file_args)
        )
        for k, v in loaded_args.items():
            if k in ignored_file_args:
                print(f"Ignoring overriding {k}")
                continue
            setattr(args, k, v)

    if args.do_overfit:
        args.dropout = 0.0
        args.dropout_heads = 0.0
        args.weight_decay = 0.0
    args.use_es_train = args.do_overfit and args.use_es
    args.use_es_val = args.use_es and not args.use_es_train
    args.use_cuda = args.device == "cuda"
    args.ds_alias = args.ds_alias or args.ds_name

    assert not (args.do_predict_6d_rot and args.do_predict_3d_rot), "Cannot predict both 6D and 3D rotation"
    assert not (args.do_predict_rel_pose and args.t_loss_name == "mixed"), "Mixed t loss is not working with rel pose"

    if args.do_predict_abs_pose:
        assert args.do_predict_rel_pose, "do_predict_abs_pose is used in conjunction with do_predict_abs_pose"

    if args.ds_name not in ["ycbi", "cube"]:
        if args.do_overfit:
            args.ds_folder_name_val = args.ds_folder_name_train
        else:
            assert args.ds_folder_name_val, "Validation dataset folder name is required for training"

    if args.exp_name.startswith("args_"):
        args.exp_name = args.exp_name.replace("args_", "")

    args.rot_out_dim = 6 if args.do_predict_6d_rot else (3 if args.do_predict_3d_rot else 4)
    args.t_out_dim = 2 if args.do_predict_2d_t else 3
    # from rotation_conversions.py
    args.rot_repr = (
        "rotation6d" if args.do_predict_6d_rot else ("axis_angle" if args.do_predict_3d_rot else "quaternion")
    )
    args.t_repr = "2d" if args.do_predict_2d_t else "3d"

    return args


def fix_outdated_args(args):
    parser = get_parser()

    def noattr(x):
        return not hasattr(args, x)

    def is_none(x):
        return getattr(args, x) is None

    if noattr("obj_names"):
        args.obj_names = [args.obj_name]
    if noattr("args_path"):
        args.args_path = None
    if noattr("do_predict_2d_t"):
        args.do_predict_2d_t = args.do_predict_2d
    if hasattr(args, "t_mlp_num_layers"):
        args.rt_mlps_num_layers = args.t_mlp_num_layers

    # for all args present in parser but not in args, set them to their default values
    for group in parser._action_groups:
        for action in group._group_actions:
            arg_name = action.dest
            if noattr(arg_name) or (arg_name == "opt_only" and is_none(arg_name)):
                setattr(args, arg_name, action.default)
            if arg_name == "opt_only" and getattr(args, arg_name) != action.default:
                args.exp_tags += [f"opt_{'_'.join(args.opt_only)}"]

    return args


def map_args_to_groups(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    """
    Organizes parsed arguments into a dictionary based on their argument groups.

    Parameters:
        parser: The argument parser with defined groups.
        args: The parsed arguments.

    Returns:
        A dictionary with argument names as keys and group names aliases as values.
    """
    arg_group_map = {}
    for group in parser._action_groups:
        group_name = group.title.lower()[0]
        for action in group._group_actions:
            arg_name = action.dest
            if hasattr(args, arg_name):
                arg_group_map[arg_name] = group_name
    return arg_group_map


def load_args_from_file(path):
    with open(path, "r") as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.UnsafeLoader))
    return args
