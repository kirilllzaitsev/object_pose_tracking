import copy
import functools
import os
import shutil
import sys
import typing as t
from collections import defaultdict
from pathlib import Path
from socket import gethostname

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    DATA_DIR,
    IS_CLUSTER,
    IS_LOCAL,
    PROJ_DIR,
    YCB_MESHES_DIR,
    YCBINEOAT_SCENE_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.ds_common import batch_seq_collate_fn, seq_collate_fn
from pose_tracking.dataset.transforms import get_transforms
from pose_tracking.models.encoders import is_param_part_of_encoders
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.artifact_utils import (
    log_artifacts,
    log_exp_meta,
    log_model_meta,
)
from pose_tracking.utils.common import get_ordered_paths, print_args
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.pipe_utils import (
    Printer,
    create_tools,
    get_model,
    get_trainer,
    get_video_ds,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm


@record
def main(args, exp_tools: t.Optional[dict] = None, args_to_group_map: t.Optional[dict] = None):

    set_seed(args.seed)
    if args.use_ddp:
        assert any(x in os.environ for x in ["SLURM_PROCID", "RANK"])
        assert any(x in os.environ for x in ["SLURM_NTASKS", "WORLD_SIZE"])

    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.is_ddp_interactive:
        rank = local_rank
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
        )
        if args.use_cuda:
            torch.cuda.set_device(local_rank)
        device = torch.device(args.device, local_rank)

        is_main_process = rank == 0
    else:
        device = torch.device(args.device)
        is_main_process = True

    if args.use_ddp:
        print(f"host: {gethostname()}, {world_size=}, {rank=}, {local_rank=}")
        if is_main_process:
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
    print_args(args, logger=logger)

    logger.info(f"{PROJ_DIR=}")
    logger.info(f"{logdir=}")
    logger.info(f"{logpath=}")

    if args.ds_name in ["ycbi", "cube"]:
        ds_video_subdirs_train = args.obj_names
        ds_video_subdirs_val = args.obj_names_val
    else:
        ds_video_subdirs_train = [
            Path(p).name for p in get_ordered_paths(DATA_DIR / args.ds_folder_name_train / "env_*")
        ]
        ds_video_subdirs_val = [Path(p).name for p in get_ordered_paths(DATA_DIR / args.ds_folder_name_val / "env_*")]

    ds_video_subdirs_train = ds_video_subdirs_train[: args.max_train_videos]
    ds_video_subdirs_val = ds_video_subdirs_val[: args.max_val_videos]

    if args.ds_name in ["ikea", "cube"]:
        ds_video_dir_train = DATA_DIR / args.ds_folder_name_train
        ds_video_dir_val = DATA_DIR / args.ds_folder_name_val
    else:
        ds_video_dir_train = YCBINEOAT_SCENE_DIR
        ds_video_dir_val = YCBINEOAT_SCENE_DIR

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
    )

    train_dataset, val_dataset = datasets["train"], datasets["val"]

    logger.info(f"{len(train_dataset)=}")
    logger.info(f"{len(val_dataset)=}")

    collate_fn = batch_seq_collate_fn if args.model_name in ["videopose", "pizza"] else seq_collate_fn
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_batch_size = args.batch_size if len(val_dataset) > 8 else len(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
    else:
        shuffle = True if not args.do_overfit else False
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        if IS_LOCAL:
            val_batch_size = 1
        else:
            val_batch_size = args.batch_size if len(val_dataset) > 8 else len(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )

    model = get_model(args).to(device)

    log_model_meta(model, exp=exp, logger=logger)

    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if args.use_cuda else None,
            broadcast_buffers=False,  # how does this affect training on diff subsets
        )
    optimizer = optim.AdamW(
        [
            {
                "params": [p for name, p in model.named_parameters() if is_param_part_of_encoders(name)],
                "lr": args.lr_encoders * np.sqrt(world_size),
            },
            {
                "params": [p for name, p in model.named_parameters() if not is_param_part_of_encoders(name)],
                "lr": args.lr * np.sqrt(world_size),
            },
        ],
        weight_decay=args.weight_decay,
    )
    if args.lrs_type == "step" or not args.use_lrs:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lrs_step_size if args.use_lrs else 1000,
            gamma=args.lrs_gamma,
            verbose=args.use_lrs,
        )
    else:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=args.lrs_gamma,
            patience=args.lrs_patience,
            threshold=args.lrs_delta,
            threshold_mode=args.lrs_threshold_mode,
            verbose=True,
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
    )

    logger.info(trainer)
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

        if args.lrs_type == "step" or not args.use_lrs:
            lr_scheduler.step()
        else:
            lr_scheduler.step(history["train"]["loss"][-1])

        # clip lr to min value 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], args.lrs_min_lr)

        if epoch % args.val_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                val_stats = trainer.loader_forward(
                    val_loader,
                    stage="val",
                )
            if is_main_process:
                printer.print_stats(val_stats, "val")
                for k, v in val_stats.items():
                    history["val"][k].append(v)

                cur_val_loss = history["val"]["loss"][-1]
                best_val_loss = min(best_val_loss, cur_val_loss)
                if args.use_es_val:
                    early_stopping(loss=cur_val_loss)

                if epoch % args.save_epoch_freq == 0 and cur_val_loss <= best_val_loss:
                    log_artifacts(artifacts, exp, logdir, epoch=epoch, suffix="best")
                    printer.saved_artifacts(epoch)

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

    if is_main_process and not args.do_overfit:
        log_artifacts(artifacts, exp, logdir, epoch, suffix="last")
        printer.saved_artifacts(epoch)

    logger.info(f"# {logdir=} {os.path.basename(logdir)}")
    logger.info(f"# {logpath=}")

    if is_main_process and not args.exp_disabled:
        logger.info(f"# Experiment finished {exp._get_experiment_url()}")

    if not external_tools and is_main_process:
        exp.end()

    if args.use_ddp:
        dist.destroy_process_group()


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
        include_mask=include_mask,
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
        res["train"] = train_dataset

    if "val" in ds_types:
        if do_overfit:
            val_dataset = get_video_ds(
                ds_video_subdirs=ds_video_subdirs_train,
                ds_name=ds_name,
                seq_len=None,
                seq_step=1,
                seq_start=seq_start,
                ds_kwargs=train_ds_kwargs,
                num_samples=num_samples,
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
                seq_len=args.seq_len if is_pizza_model else None,
                seq_step=args.seq_step if is_pizza_model else 1,
                seq_start=0,
                ds_kwargs=val_ds_kwargs,
                num_samples=num_samples,
                do_preload=True,
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
        run()
