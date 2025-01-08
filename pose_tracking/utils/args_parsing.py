import argparse
import re
import sys

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
    pipe_args.add_argument("--do_print_seq_stats", action="store_true", help="Print sequence-level stats")
    pipe_args.add_argument(
        "--do_ignore_file_args_with_provided", action="store_true", help="Ignore file args if provided via CLI"
    )
    pipe_args.add_argument("--exp_tags", nargs="*", default=[], help="Tags for the experiment to log.")
    pipe_args.add_argument("--exp_name", type=str, default="test", help="Name of the experiment.")
    pipe_args.add_argument("--device", type=str, default="cuda", help="Device to use")
    pipe_args.add_argument("--args_path", type=str, help="Path to a yaml file with arguments")
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
        choices=["mse", "mae", "huber", "huber_norm", "angle", "mixed"],
    )
    train_args.add_argument(
        "--rot_loss_name",
        type=str,
        default="mse",
        help="Rotation loss name",
        choices=["geodesic", "mse", "mae", "huber", "videopose", "displacement", "geodesic_mat"],
    )
    train_args.add_argument(
        "--opt_only",
        nargs="*",
        help="List of tasks to optimize",
        default=["rot", "t", "labels", "boxes"],
        choices=["rot", "t", "labels", "boxes"],
    )

    poseformer_args = parser.add_argument_group("PoseFormer arguments")
    poseformer_args.add_argument("--mt_do_calibrate_kpt", action="store_true", help="Calibrate keypoints")
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

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument(
        "--do_predict_2d_t", action="store_true", help="Predict object 2D center and depth separately"
    )
    model_args.add_argument("--do_predict_6d_rot", action="store_true", help="Predict object rotation as 6D")
    model_args.add_argument("--do_predict_3d_rot", action="store_true", help="Predict object rotation as 3D")
    model_args.add_argument("--do_predict_rel_pose", action="store_true", help="Predict relative pose")
    model_args.add_argument("--do_predict_kpts", action="store_true", help="Predict keypoints")
    model_args.add_argument("--use_prev_latent", action="store_true", help="Use t-1 latent as condition")
    model_args.add_argument("--no_rnn", action="store_true", help="Use a simple MLP instead of RNN")
    model_args.add_argument("--use_priv_decoder", action="store_true", help="Use privileged info decoder")
    model_args.add_argument("--do_freeze_encoders", action="store_true", help="Whether to freeze encoder backbones")
    model_args.add_argument("--use_prev_pose_condition", action="store_true", help="Use previous pose as condition")
    model_args.add_argument(
        "--no_obs_belief", action="store_true", help="Do not use observation belief encoder-decoder"
    )
    model_args.add_argument(
        "--model_name",
        type=str,
        default="cnnlstm",
        help="Model name",
        choices=["cnnlstm", "cnnlstm_sep", "videopose", "detr", "detr_basic", "detr_kpt", "pizza", "trackformer"],
    )
    model_args.add_argument(
        "--rnn_type", type=str, default="gru", help="RNN type", choices=["gru", "lstm", "gru_custom", "lstm_custom"]
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
    model_args.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the model")

    data_args = parser.add_argument_group("Data arguments")
    data_args.add_argument("--do_preload_ds", action="store_true", help="Preload videos")
    data_args.add_argument("--seq_len", type=int, default=5, help="Number of frames to take for train/val")
    data_args.add_argument("--seq_len_test", type=int, default=600, help="Number of frames to take for test")
    data_args.add_argument("--seq_start", type=int, help="Start frame index in a sequence")
    data_args.add_argument("--seq_step", type=int, default=1, help="Step between frames in a sequence")
    data_args.add_argument("--max_train_videos", type=int, default=1000, help="Max number of videos for training")
    data_args.add_argument("--max_val_videos", type=int, default=100, help="Max number of videos for validation")
    data_args.add_argument("--num_samples", type=int, help="Number of times to fetch sequences (len(video_ds))")
    data_args.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    data_args.add_argument("--obj_names", nargs="*", default=[], help="Object names to use in the dataset")
    data_args.add_argument(
        "--obj_names_val",
        nargs="*",
        default=[],
        help="Object names to use in the validation dataset",
    )
    data_args.add_argument(
        "--ds_name", type=str, default="ycbi", help="Dataset name", choices=["ycbi", "cube_sim", "ikea"]
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
        args.weight_decay = 0.0
    args.use_es_train = args.do_overfit and args.use_es
    args.use_es_val = args.use_es and not args.use_es_train
    args.use_cuda = args.device == "cuda"

    assert not (args.do_predict_6d_rot and args.do_predict_3d_rot), "Cannot predict both 6D and 3D rotation"
    assert not (args.do_predict_rel_pose and args.t_loss_name == "mixed"), "Mixed t loss is not working with rel pose"

    if args.ds_name not in ["ycbi", "cube"]:
        if args.do_overfit:
            args.ds_folder_name_val = args.ds_folder_name_train
        else:
            assert args.ds_folder_name_val, "Validation dataset folder name is required for training"

    if args.exp_name.startswith("args_"):
        args.exp_name = args.exp_name.replace("args_", "")

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
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    return args
