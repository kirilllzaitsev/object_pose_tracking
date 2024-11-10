import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    pipe_args = parser.add_argument_group("Training arguments")

    pipe_args.add_argument("--exp_tags", nargs="*", default=[], help="Tags for the experiment to log.")
    pipe_args.add_argument("--exp_name", type=str, default="test", help="Name of the experiment.")
    pipe_args.add_argument("--exp_disabled", action="store_true", help="Disable experiment logging.")
    pipe_args.add_argument("--do_overfit", action="store_true", help="Overfit setting")
    pipe_args.add_argument("--do_debug", action="store_true", help="Debugging setting")

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    train_args.add_argument("--val_epoch_freq", type=int, default=5, help="Validate every N epochs")
    train_args.add_argument("--save_epoch_freq", type=int, default=10, help="Save model every N epochs")
    train_args.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    train_args.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    train_args.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    train_args.add_argument("--seed", type=int, default=10, help="Random seed")
    train_args.add_argument("--use_early_stopping", action="store_true", help="Use early stopping")
    train_args.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument(
        "--rnn_type", type=str, default="gru", help="RNN type", choices=["gru", "lstm", "gru_custom"]
    )
    model_args.add_argument(
        "--encoder_name", type=str, default="regnet_y_800mf", help="Encoder name for both RGB and depth"
    )
    model_args.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension across the model")
    model_args.add_argument(
        "--bdec_priv_decoder_hidden_dim", type=int, default=40, help="Hidden dimension for privilaged info decoder"
    )
    model_args.add_argument(
        "--bdec_depth_decoder_hidden_dim", type=int, default=80, help="Hidden dimension for depth decoder"
    )
    model_args.add_argument(
        "--benc_belief_enc_hidden_dim", type=int, default=110, help="Hidden dimension for belief encoder"
    )
    model_args.add_argument(
        "--benc_belief_depth_enc_hidden_dim", type=int, default=200, help="Hidden dimension for belief depth encoder"
    )
    model_args.add_argument(
        "--bdec_hidden_attn_hidden_dim", type=int, default=128, help="Hidden dimension for hidden attention"
    )

    data_args = parser.add_argument_group("Data arguments")
    data_args.add_argument("--scene_idx", type=int, default=7, help="Scene index")
    data_args.add_argument("--seq_length", type=int, default=3, help="Number of frames to take")
    data_args.add_argument("--seq_start", type=int, help="Start frame index in a sequence")
    data_args.add_argument("--seq_step", type=int, default=1, help="Step between frames in a sequence")
    data_args.add_argument("--num_samples", type=int, help="Number of sequence frames to take")
    data_args.add_argument("--ds_name", type=str, default="lm", help="Dataset name", choices=["lm", "ycbv", "ycbi"])
    data_args.add_argument("--obj_name", type=str, default="mustard0", help="Object name in the dataset")
    data_args.add_argument("--use_video_ds", action="store_true", help="Use video dataset")

    args_raw = parser.parse_args()
    args = postprocess_args(args_raw)
    return args


def postprocess_args(args):
    if args.do_overfit:
        args.dropout_prob = 0.0

    return args
