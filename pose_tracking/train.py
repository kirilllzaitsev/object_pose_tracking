import functools
import json
import os
import shutil
import sys
import traceback
import typing as t
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from socket import gethostname

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from comet_ml.integration.pytorch import watch
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    DATA_DIR,
    HO3D_ROOT,
    IS_CLUSTER,
    PROJ_DIR,
    YCBINEOAT_SCENE_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.ds_common import batch_seq_collate_fn, seq_collate_fn
from pose_tracking.dataset.ds_meta import YCBV_OBJ_NAME_TO_ID
from pose_tracking.models.encoders import is_param_part_of_encoders
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.artifact_utils import (
    log_artifacts,
    log_exp_meta,
    log_model_meta,
)
from pose_tracking.utils.comet_utils import load_artifacts_from_comet, log_params_to_exp
from pose_tracking.utils.common import get_ordered_paths, print_args
from pose_tracking.utils.misc import print_error_locals, set_seed
from pose_tracking.utils.pipe_utils import (
    Printer,
    create_tools,
    get_datasets,
    get_ds_dirs,
    get_model,
    get_num_classes,
    get_trainer,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


@record
def main(args, exp_tools: t.Optional[dict] = None, args_to_group_map: t.Optional[dict] = None):

    set_seed(args.seed)
    if args.use_ddp:
        assert any(x in os.environ for x in ["SLURM_PROCID", "RANK"])
        assert any(x in os.environ for x in ["SLURM_NTASKS", "WORLD_SIZE"])

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.use_ddp:
        print(f"host: {gethostname()}, {world_size=}, {rank=}, {local_rank=}")
    if args.use_ddp:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(np.random.randint(20000, 30000))

        dist.init_process_group(
            backend="nccl" if args.use_cuda else "gloo",
            world_size=world_size,
            init_method="env://",
            rank=rank,
            timeout=dt.timedelta(seconds=1800),
        )
        if args.use_cuda:
            torch.cuda.set_device(local_rank)
        device = torch.device(args.device, local_rank)

        is_main_process = rank == 0
    else:
        device = torch.device(args.device)
        is_main_process = True

    if args.use_ddp and is_main_process:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    if "SLURM_GPUS_ON_NODE" in os.environ:
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert gpus_per_node == torch.cuda.device_count()

    external_tools = True
    if exp_tools is None:
        external_tools = False
        if is_main_process:
            exp_tools = create_tools(args)

        else:
            exp_tools = defaultdict(lambda: None)
    logdir = exp_tools["logdir"]
    exp = exp_tools["exp"]
    writer = exp_tools["writer"]
    if is_main_process:
        log_exp_meta(args, save_args=True, logdir=logdir, exp=exp, args_to_group_map=args_to_group_map)

    logpath = f"{logdir}/log.log"
    logger = prepare_logger(logpath=logpath, level="INFO")
    if is_main_process:
        sys.excepthook = log_exception
    else:
        logger.remove()
    logger.info(f"CLI command:\npython {' '.join(sys.argv)}")
    if "SLURM_JOB_ID" in os.environ:
        logger.info(f"SLURM_JOB_ID: {os.environ['SLURM_JOB_ID']}")

    ds_dirs = get_ds_dirs(args)
    ds_video_dir_train = ds_dirs["ds_video_dir_train"]
    ds_video_dir_val = ds_dirs["ds_video_dir_val"]
    ds_video_subdirs_train = ds_dirs["ds_video_subdirs_train"]
    ds_video_subdirs_val = ds_dirs["ds_video_subdirs_val"]

    if args.num_classes is None:
        num_classes = get_num_classes(args, ds_video_dir_train)
        args.num_classes = num_classes
        logger.info(f"{num_classes=}")
        if is_main_process:
            log_params_to_exp(
                exp,
                {"num_classes": num_classes},
                "args",
            )
    else:
        num_classes = args.num_classes

    print_args(args, logger=logger)
    logger.info(f"{PROJ_DIR=}")
    logger.info(f"{logdir=}")
    logger.info(f"{logpath=}")

    datasets = get_datasets(
        ds_name=args.ds_name,
        seq_len=args.seq_len,
        seq_step=args.seq_step,
        seq_start=args.seq_start,
        ds_video_dir_train=ds_video_dir_train,
        ds_video_dir_val=ds_video_dir_val,
        ds_video_subdirs_train=ds_video_subdirs_train,
        ds_video_subdirs_val=ds_video_subdirs_val,
        transform_names=args.transform_names,
        transform_prob=args.transform_prob,
        mask_pixels_prob=args.mask_pixels_prob,
        num_samples=args.num_samples,
        do_predict_kpts=args.do_predict_kpts,
        use_priv_decoder=args.use_priv_decoder,
        do_overfit=args.do_overfit,
        do_preload_ds=args.do_preload_ds,
        model_name=args.model_name,
        max_train_videos=args.max_train_videos,
        max_val_videos=args.max_val_videos,
        end_frame_idx=args.end_frame_idx,
        rot_repr=args.rot_repr,
        t_repr=args.t_repr,
        max_random_seq_step=args.max_random_seq_step,
        do_predict_rel_pose=args.do_predict_rel_pose,
        do_subtract_bg=args.do_subtract_bg,
        use_entire_seq_in_train=args.use_entire_seq_in_train,
        seq_len_max_train=args.seq_len_max_train,
        include_depth=args.mt_use_depth,
        include_mask=args.mt_use_mask_on_input,
        include_bbox_2d=args.use_crop_for_rot,
        do_normalize_depth=args.mt_use_depth,
        bbox_num_kpts=args.bbox_num_kpts,
        dino_features_folder_name=args.dino_features_folder_name,
        use_mask_for_bbox_2d=args.use_mask_for_bbox_2d,
        factors=args.factors,
    )

    train_dataset, val_dataset, train_as_val_dataset = (
        datasets["train"],
        datasets["val"],
        datasets["train_as_val"],
    )

    logger.info(f"{ds_video_dir_train=}")
    logger.info(f"{ds_video_dir_val=}")
    logger.info(f"{len(ds_video_subdirs_train)=}")
    logger.info(f"{len(ds_video_subdirs_val)=}")
    logger.info(f"{len(train_dataset)=}")
    logger.info(f"{len(val_dataset)=}")
    logger.info(f"{len(train_as_val_dataset)=}")
    logger.info(f"{train_dataset.video_datasets[0]=}")
    logger.info(f"{val_dataset.video_datasets[0]=}")
    logger.info(f"{train_as_val_dataset.video_datasets[0]=}")

    if is_main_process:
        log_params_to_exp(
            exp,
            {
                "len_train_videos": len(train_dataset.video_datasets),
                "len_val_videos": len(val_dataset.video_datasets),
                "len_train_as_val_videos": len(train_as_val_dataset.video_datasets),
            },
            "d",
        )

    collate_fn = batch_seq_collate_fn if args.model_name in ["videopose", "pizza", "motr", "memotr"] else seq_collate_fn

    val_batch_size = min(max(1, args.num_workers), args.batch_size)
    if "four_large" in str(ds_video_dir_train):
        val_batch_size = max(1, val_batch_size // 4)
    if "memotr" in args.model_name:
        val_batch_size = 1

    logger.info(f"{args.batch_size=}")
    logger.info(f"{val_batch_size=}")

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        train_as_val_sampler = DistributedSampler(
            train_as_val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
        )
        train_as_val_loader = DataLoader(
            train_as_val_dataset,
            batch_size=val_batch_size,
            sampler=train_as_val_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
    else:
        shuffle = False if args.do_overfit else True
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=True,
        )
        train_as_val_loader = DataLoader(
            train_as_val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=True,
        )

    model = get_model(args, num_classes=num_classes).to(device)

    log_model_meta(model, exp=exp, logger=logger)
    logger.info(model)

    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if args.use_cuda else None,
            broadcast_buffers=False,  # how does this affect training on diff subsets
        )

    trainer = get_trainer(
        args,
        model,
        device=device,
        writer=writer,
        world_size=world_size,
        logger=logger,
        do_vis=args.do_vis and is_main_process,
        exp_dir=logdir,
        num_classes=num_classes,
    )

    if any(x in args.model_name for x in ["cnnlstm", "pizza", "cvae", "cnn", "kpt_pose", "cnn_kpt", "kpt_kpt"]):
        optimizer = optim.AdamW(
            [
                {
                    "params": [p for name, p in model.named_parameters() if is_param_part_of_encoders(name)],
                    "lr": args.lr_encoders,
                },
                {
                    "params": [p for name, p in model.named_parameters() if not is_param_part_of_encoders(name)],
                    "lr": args.lr,
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = trainer.optimizer

    if args.lrs_type == "step" or not args.use_lrs:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lrs_step_size if args.use_lrs else 1000,
            gamma=args.lrs_gamma,
        )
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=args.lrs_gamma,
            patience=args.lrs_patience,
            threshold=args.lrs_delta,
            threshold_mode=args.lrs_threshold_mode,
            min_lr=args.lrs_min_lr,
        )
    if args.ckpt_exp_name:
        download_res = load_artifacts_from_comet(exp_name=args.ckpt_exp_name, do_load_session=True, artifact_suffix="best")
        if download_res["session_checkpoint_path"] is None:
            logger.warning(f"Checkpoint not found for {args.ckpt_exp_name}")
        else:
            opt_state = torch.load(download_res["session_checkpoint_path"])
            optimizer.load_state_dict(opt_state["optimizer"])
            lr_scheduler.load_state_dict(opt_state["scheduler"])
            logger.warning(f"Loaded optimizer state from {args.ckpt_exp_name}")

    logger.info(trainer)
    if logdir is not None:
        logger.info(f"{logdir=} {os.path.basename(logdir)}")
    if is_main_process and not args.exp_disabled:
        logger.info(f"# Experiment created at {exp._get_experiment_url()}")
        logger.info(f'# Please leave a note about the experiment at {exp._get_experiment_url(tab="notes")}')

    early_stopping = EarlyStopping(patience=args.es_patience_epochs, delta=args.es_delta, verbose=True)
    artifacts = {
        "model": model.module if args.use_ddp else model,
        "optimizer": optimizer,
        "scheduler": lr_scheduler,
    }
    printer = Printer(logger)
    best_val_loss = np.inf
    history = defaultdict(lambda: defaultdict(list))
    if is_main_process:
        log_model_stats_epoch_freq = 15 if args.do_overfit else 5
        steps_per_epoch = len(train_loader) * (1 if args.use_rnn else args.seq_len) * world_size
        watch(model, log_step_interval=steps_per_epoch * log_model_stats_epoch_freq)

    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
        model.train()
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        train_stats = trainer.loader_forward(
            train_loader,
            optimizer=optimizer,
            stage="train",
        )

        logger.info(f"# Epoch {epoch} #")
        printer.print_stats(train_stats, "train")
        for k, v in train_stats.items():
            history["train"][k].append(v)

        if (epoch - 1) % args.val_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                val_stats = trainer.loader_forward(
                    val_loader,
                    stage="val",
                )
                train_as_val_stats = (
                    {}
                    if args.do_overfit
                    else trainer.loader_forward(
                        train_as_val_loader,
                        stage="train_as_val",
                    )
                )
            if is_main_process:
                for stage, stats in zip(["val", "train_as_val"], [val_stats, train_as_val_stats]):
                    printer.print_stats(stats, stage)
                    for k, v in stats.items():
                        history[stage][k].append(v)

                cur_val_loss = history["val"]["loss"][-1]
                best_val_loss = min(best_val_loss, cur_val_loss)
                if args.use_es_val:
                    early_stopping(loss=cur_val_loss)

                if args.do_save_artifacts and epoch % args.save_epoch_freq == 0 and cur_val_loss <= best_val_loss:
                    log_artifacts(artifacts, exp, logdir, epoch=epoch, suffix="best", do_log_session=True)
                    printer.saved_artifacts(epoch)

                last_lrs_before = lr_scheduler.get_last_lr()
                if args.lrs_type == "step" or not args.use_lrs:
                    lr_scheduler.step()
                else:
                    lr_scheduler.step(history["val"]["loss"][-1])
                last_lrs_after = lr_scheduler.get_last_lr()
                for param_group in optimizer.param_groups:
                    param_group["lr"] = max(param_group["lr"], args.lrs_min_lr)
                for gidx, (lr_before, lr_after) in enumerate(zip(last_lrs_before, last_lrs_after)):
                    if lr_before != lr_after and lr_after > args.lrs_min_lr:
                        logger.warning(f"Changing lr from {lr_before} to {lr_after} for {gidx=}")

        if args.use_es_train and is_main_process:
            early_stopping(loss=history["train"]["loss"][-1])

        if is_main_process:
            for i, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group_{i}", pg["lr"], epoch)
            writer.add_scalar("epoch", epoch, epoch)

        if early_stopping.do_stop:
            logger.warning(f"Early stopping on epoch {epoch}")
            break

    if args.do_debug:
        shutil.rmtree(logdir)
        return

    if is_main_process and args.do_save_artifacts:
        log_artifacts(artifacts, exp, logdir, epoch, suffix="last", do_log_session=True)
        printer.saved_artifacts(epoch)

    if logdir is not None:
        logger.info(f"# {logdir=} {os.path.basename(logdir)}")
    logger.info(f"# {logpath=}")

    if is_main_process and not args.exp_disabled:
        logger.info(f"# Experiment finished {exp._get_experiment_url()}")

    if not external_tools and is_main_process:
        exp.end()

    if args.use_ddp:
        dist.destroy_process_group()

    return history


if __name__ == "__main__":

    if not IS_CLUSTER:
        import matplotlib

        matplotlib.use("TkAgg")

    args, args_to_group_map = parse_args()
    import datetime as dt

    from pose_tracking.utils.profiling_utils import profile_func

    run = functools.partial(main, args=args, args_to_group_map=args_to_group_map)
    if args.do_profile:
        now = dt.datetime.now()
        profile_func(run, f"{PROJ_DIR}/profiling/{now.strftime('%Y%m%d_%H%M%S')}.prof")
    else:
        try:
            run()
        except Exception as e:
            print_error_locals()
            traceback.print_exc()
            raise e
