import copy
import os
import shutil
import sys
import typing as t
from collections import defaultdict

import comet_ml
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    DATA_DIR,
    PROJ_DIR,
    YCB_MESHES_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.ds_common import batch_seq_collate_fn, seq_collate_fn
from pose_tracking.dataset.transforms import get_transforms
from pose_tracking.models.encoders import is_param_part_of_encoders
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.common import print_args
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.pipe_utils import (
    Printer,
    create_tools,
    get_full_ds,
    get_model,
    get_trainer,
    log_artifacts,
    log_exp_meta,
    log_model_meta,
    print_stats,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm


@record
def main(exp_tools: t.Optional[dict] = None):
    args, args_to_group_map = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.use_ddp:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(np.random.randint(20000, 30000))

        dist.init_process_group(
            backend="nccl" if args.use_cuda else "gloo", init_method="env://", world_size=world_size, rank=rank
        )
        if args.use_cuda:
            torch.cuda.set_device(local_rank)
        device = torch.device(args.device, local_rank)

        is_main_process = rank == 0
    else:
        is_main_process = True

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

    if is_main_process and not args.exp_disabled:
        logger.info(f"# Experiment created at {exp._get_experiment_url()}")
        logger.info(f'# Please leave a note about the experiment at {exp._get_experiment_url(tab="notes")}')

    logger.info(f"{PROJ_DIR=}")
    logger.info(f"{logdir=}")
    logger.info(f"{logpath=}")

    if args.use_ddp:
        print(
            f"Hello from rank {rank} of {world_size - 1} where there are {world_size} allocated GPUs per node.",
        )

    transform = get_transforms(args.transform_names, transform_prob=args.transform_prob)
    if args.ds_name == "ycbi":
        ycbi_kwargs = dict(
            shorter_side=None,
            zfar=np.inf,
            include_rgb=True,
            include_depth=True,
            include_gt_pose=True,
            include_mask=True,
            include_bbox_2d=True if args.model_name in ["cnnlstm_sep"] else False,
            ycb_meshes_dir=YCB_MESHES_DIR,
            transforms_rgb=transform,
            start_frame_idx=0,
            convert_pose_to_quat=True,
            mask_pixels_prob=args.mask_pixels_prob,
        )
        ds_kwargs = ycbi_kwargs
    else:
        sim_ds_path = DATA_DIR / args.ds_folder_name
        cube_sim_kwargs = dict(
            root_dir=f"{sim_ds_path}",
            mesh_path=f"{sim_ds_path}/mesh/cube.obj",
            include_masks=True,
            use_priv_info=args.use_priv_decoder,
            convert_pose_to_quat=True,
        )
        ds_kwargs = cube_sim_kwargs
    full_ds = get_full_ds(
        obj_names=args.obj_names,
        ds_name=args.ds_name,
        seq_len=args.seq_len,
        seq_step=args.seq_step,
        seq_start=args.seq_start,
        num_samples=args.num_samples,
        ds_kwargs=ds_kwargs,
    )
    scene_len = len(full_ds)
    logger.info(f"Scene length: {scene_len}")

    train_dataset = full_ds

    val_ds_kwargs = copy.deepcopy(ds_kwargs)
    val_ds_kwargs.pop("mask_pixels_prob")
    val_dataset = get_full_ds(
        obj_names=args.obj_names_val,
        ds_name=args.ds_name,
        seq_len=args.seq_len,
        seq_step=1,
        seq_start=None,
        num_samples=min(args.num_samples, 500),
        ds_kwargs=val_ds_kwargs,
    )

    logger.info(f"Using {args.obj_names_val=} for validation")

    if args.do_overfit:
        val_dataset = train_dataset

    logger.info(f"{len(train_dataset)=}")
    logger.info(f"{len(val_dataset)=}")

    collate_fn = batch_seq_collate_fn if args.model_name in ["videopose"] else seq_collate_fn
    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
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
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
        )

    model = get_model(args).to(device)

    log_model_meta(model, exp=exp, logger=logger)

    if args.use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank] if args.use_cuda else None,
            output_device=local_rank if args.use_cuda else None,
            broadcast_buffers=False,  # how does this affect training on diff subs
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
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lrs_step_size, gamma=args.lrs_gamma)

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
        print_stats(train_stats, logger, "train")
        for k, v in train_stats.items():
            history["train"][k].append(v)

        lr_scheduler.step()

        # clip lr to min value 1e-6
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(param_group["lr"], 1e-6)

        if epoch % args.val_epoch_freq == 0 and not args.do_overfit:
            model.eval()
            with torch.no_grad():
                val_stats = trainer.loader_forward(
                    val_loader,
                    stage="val",
                )
            if is_main_process:
                print_stats(val_stats, logger, "val")
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

    logger.info(f"# {logdir=}")
    logger.info(f"# {logpath=}")

    if not external_tools and is_main_process:
        exp.end()

    if args.use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
