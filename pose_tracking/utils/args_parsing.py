import argparse

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
    pipe_args.add_argument("--exp_tags", nargs="*", default=[], help="Tags for the experiment to log.")
    pipe_args.add_argument("--exp_name", type=str, default="test", help="Name of the experiment.")
    pipe_args.add_argument("--device", type=str, default="cuda", help="Device to use")
    pipe_args.add_argument("--args_path", type=str, help="Path to a yaml file with arguments")
    pipe_args.add_argument(
        "--ignored_file_args",
        nargs="*",
        default=[],
        help="List of args to ignore when loading from file.",
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
    train_args.add_argument("--vis_epoch_freq", type=int, default=5, help="Visualize a random sequence every N epochs")
    train_args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    train_args.add_argument("--val_epoch_freq", type=int, default=1, help="Validate every N epochs")
    train_args.add_argument("--save_epoch_freq", type=int, default=1, help="Save model every N epochs")
    train_args.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    train_args.add_argument("--seed", type=int, default=10, help="Random seed")
    train_args.add_argument("--es_patience_epochs", type=int, default=3, help="Early stopping patience")
    train_args.add_argument("--es_delta", type=float, default=1e-3, help="Early stopping delta")
    train_args.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_args.add_argument("--lr_encoders", type=float, default=1e-5, help="Learning rate")
    train_args.add_argument("--lrs_step_size", type=int, default=10, help="Number of epochs before changing lr")
    train_args.add_argument("--lrs_gamma", type=float, default=0.5, help="Scaler for learning rate scheduler")
    train_args.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
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
        default="geodesic",
        help="Rotation loss name",
        choices=["geodesic", "mse", "mae", "huber", "videopose"],
    )

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
        choices=["cnnlstm", "cnnlstm_sep", "videopose", "detr", "detr_basic"],
    )
    model_args.add_argument(
        "--rnn_type", type=str, default="gru", help="RNN type", choices=["gru", "lstm", "gru_custom", "lstm_custom"]
    )
    model_args.add_argument(
        "--encoder_name", type=str, default="regnet_y_800mf", help="Encoder name for both RGB and depth"
    )
    model_args.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension across the model")
    model_args.add_argument(
        "--benc_belief_enc_hidden_dim", type=int, default=512, help="Hidden dimension for belief encoder"
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
        "--priv_decoder_num_layers", type=int, default=1, help="Number of layers for privileged info decoder"
    )
    model_args.add_argument(
        "--depth_decoder_num_layers", type=int, default=1, help="Number of layers for depth decoder from hidden state"
    )
    model_args.add_argument(
        "--hidden_attn_num_layers", type=int, default=1, help="Number of layers for hidden attention in decoder"
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
    data_args.add_argument("--num_samples", type=int, help="Number of sequence frames to take")
    data_args.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    data_args.add_argument("--obj_names", nargs="*", default=["mustard0"], help="Object names to use in the dataset")
    data_args.add_argument(
        "--obj_names_val",
        nargs="*",
        default=["mustard_easy_00_02"],
        help="Object names to use in the validation dataset",
    )
    data_args.add_argument(
        "--ds_name", type=str, default="ycbi", help="Dataset name", choices=["ycbi", "cube_sim", "ikea"]
    )
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


def postprocess_args(args):

    args = fix_outdated_args(args)

    if args.args_path:
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
        ignored_file_args = set(args.ignored_file_args) | set(default_ignored_file_args)
        for k, v in loaded_args.items():
            if k in ignored_file_args:
                print(f"Ignoring overriding {k}")
                continue
            setattr(args, k, v)

    if args.do_overfit:
        args.dropout = 0.0
    args.use_es_train = args.do_overfit and args.use_es
    args.use_es_val = args.use_es and not args.use_es_train
    args.use_cuda = args.device == "cuda"

    assert not (args.do_predict_6d_rot and args.do_predict_3d_rot), "Cannot predict both 6D and 3D rotation"
    if args.ds_name not in ["ycbi", "cube"]:
        if args.do_overfit:
            args.ds_folder_name_val = args.ds_folder_name_train
        else:
            assert args.ds_folder_name_val, "Validation dataset folder name is required for training"

    return args


def fix_outdated_args(args):
    parser = get_parser()

    def noattr(x):
        return not hasattr(args, x)

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
            if noattr(arg_name):
                setattr(args, arg_name, action.default)

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
