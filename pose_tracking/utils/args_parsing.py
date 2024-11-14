import argparse

import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("Training arguments")

    pipe_args.add_argument("--exp_tags", nargs="*", default=[], help="Tags for the experiment to log.")
    pipe_args.add_argument("--exp_name", type=str, default="test", help="Name of the experiment.")
    pipe_args.add_argument("--exp_disabled", action="store_true", help="Disable experiment logging.")
    pipe_args.add_argument("--do_overfit", action="store_true", help="Overfit setting")
    pipe_args.add_argument("--do_debug", action="store_true", help="Debugging setting")
    pipe_args.add_argument("--use_test_set", action="store_true", help="Predict on a test set")
    pipe_args.add_argument("--device", type=str, default="cuda", help="Device to use")
    pipe_args.add_argument("--args_path", type=str, help="Path to a yaml file with arguments")
    pipe_args.add_argument(
        "--ignored_file_args",
        nargs="*",
        default=[],
        help="List of args to ignore when loading from file.",
    )

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    train_args.add_argument("--val_epoch_freq", type=int, default=5, help="Validate every N epochs")
    train_args.add_argument("--save_epoch_freq", type=int, default=10, help="Save model every N epochs")
    train_args.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    train_args.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    train_args.add_argument("--seed", type=int, default=10, help="Random seed")
    train_args.add_argument("--use_es", action="store_true", help="Use early stopping")
    train_args.add_argument("--es_patience", type=int, default=10, help="Early stopping patience")
    train_args.add_argument("--es_delta", type=float, default=1e-3, help="Early stopping delta")
    train_args.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    train_args.add_argument("--lrs_step_size", type=int, default=10, help="Number of epochs before changing lr")
    train_args.add_argument("--lrs_gamma", type=float, default=0.5, help="Scaler for learning rate scheduler")
    train_args.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    train_args.add_argument(
        "--pose_loss_name", type=str, default="add", help="Pose loss name", choices=["separate", "add"]
    )

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument(
        "--do_predict_2d", action="store_true", help="Predict object 2D center and depth separately"
    )
    model_args.add_argument("--do_predict_6d_rot", action="store_true", help="Predict object rotation as 6D")
    model_args.add_argument(
        "--rnn_type", type=str, default="gru", help="RNN type", choices=["gru", "lstm", "gru_custom"]
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
        "--bdec_priv_decoder_hidden_dim", type=int, default=32, help="Hidden dimension for privileged info decoder"
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
    data_args.add_argument("--seq_len", type=int, default=5, help="Number of frames to take for train/val")
    data_args.add_argument("--seq_len_test", type=int, help="Number of frames to take for test")
    data_args.add_argument("--seq_start", type=int, help="Start frame index in a sequence")
    data_args.add_argument("--seq_step", type=int, default=1, help="Step between frames in a sequence")
    data_args.add_argument("--num_samples", type=int, help="Number of sequence frames to take")
    data_args.add_argument("--obj_names", nargs="+", default=["mustard0"], help="Object names to use in the dataset")

    args_raw, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"WARNING. Unknown arguments: {unknown_args}")
    args = postprocess_args(args_raw)
    args_to_group_map = map_args_to_groups(parser, args)
    return args, args_to_group_map


def postprocess_args(args):

    args = fix_outdated_args(args)

    if args.args_path:
        import yaml

        with open(args.args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.FullLoader)

        default_ignored_file_args = [
            "device",
            "run_name",
            "log_subdir",
            "ddp",
            "exp_disabled",
            "batch_size",
            "vis_epoch_freq",
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

    return args


def fix_outdated_args(args):
    if not hasattr(args, "obj_names"):
        args.obj_names = [args.obj_name]
    if not hasattr(args, "args_path"):
        args.args_path = None

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
