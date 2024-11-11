import os
import shutil
import sys
import typing as t
from collections import defaultdict
from pathlib import Path

import comet_ml
import cv2
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    PROJ_DIR,
    YCB_MESHES_DIR,
    YCBINEOAT_SCENE_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.dataset.ds_common import seq_collate_fn
from pose_tracking.dataset.transforms import get_transforms
from pose_tracking.dataset.video_ds import VideoDataset
from pose_tracking.dataset.ycbineoat import YCBineoatDataset
from pose_tracking.losses import compute_add_loss, geodesic_loss
from pose_tracking.metrics import calc_metrics
from pose_tracking.models.cnnlstm import RecurrentCNN
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.common import adjust_img_for_plt, cast_to_numpy, print_args
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.pipe_utils import create_tools, log_exp_meta
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

matplotlib.use("TKAgg")


def main(exp_tools: t.Optional[dict] = None):
    args, args_to_group_map = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    if args.ddp:
        rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

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
    model_path = f"{logdir}/model.pth"

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
    print_args(args, logger=logger)
    logger.info(f"{PROJ_DIR=}")
    logger.info(f"{logdir=}")
    logger.info(f"{logpath=}")

    if is_main_process and not args.exp_disabled:
        logger.info(f"Experiment created at {exp._get_experiment_url()}")
        logger.info(f'Please leave a note about the experiment at {exp._get_experiment_url(tab="notes")}')

    if args.ddp:
        print(
            f"Hello from rank {rank} of {world_size - 1} where there are {world_size} allocated GPUs per node.",
        )

    criterion_trans = nn.MSELoss()
    criterion_rot = geodesic_loss
    criterion_pose = compute_add_loss if args.pose_loss_name in ["add"] else None

    transform = get_transforms()

    early_stopping = EarlyStopping(patience=args.es_patience, delta=args.es_delta, verbose=True)

    ycbi_kwargs = dict(
        video_dir=YCBINEOAT_SCENE_DIR / args.obj_name,
        shorter_side=None,
        zfar=np.inf,
        include_rgb=True,
        include_depth=True,
        include_gt_pose=True,
        include_mask=True,
        ycb_meshes_dir=YCB_MESHES_DIR,
        transforms=transform,
        start_frame_idx=70,
        convert_pose_to_quat=True,
    )
    ds_ycbi = YCBineoatDataset(**ycbi_kwargs)
    video_ds = VideoDataset(
        ds=ds_ycbi,
        seq_len=args.seq_len,
        seq_step=args.seq_step,
        seq_start=args.seq_start,
        num_samples=args.num_samples,
    )
    full_ds = video_ds
    scene_len = len(full_ds)
    logger.info(f"Scene length: {scene_len}")
    train_share = 1.0 if args.do_overfit else 0.8
    train_len = int(train_share * scene_len)

    train_dataset = torch.utils.data.Subset(full_ds, range(train_len))
    val_dataset = torch.utils.data.Subset(full_ds, range(train_len, scene_len))

    if args.do_overfit:
        val_dataset = train_dataset

    logger.info(f"{len(train_dataset)=}")

    collate_fn = seq_collate_fn
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn
        )
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
    else:
        shuffle = True if not args.do_overfit else False
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    history = defaultdict(lambda: defaultdict(list))

    hidden_dim = args.hidden_dim
    priv_dim = 1
    latent_dim = 256
    depth_dim = latent_dim
    rgb_dim = latent_dim
    model = RecurrentCNN(
        depth_dim=depth_dim,
        rgb_dim=rgb_dim,
        hidden_dim=hidden_dim,
        rnn_type=args.rnn_type,
        bdec_priv_decoder_out_dim=priv_dim,
        bdec_priv_decoder_hidden_dim=args.bdec_priv_decoder_hidden_dim,
        bdec_depth_decoder_hidden_dim=args.bdec_depth_decoder_hidden_dim,
        benc_belief_enc_hidden_dim=args.benc_belief_enc_hidden_dim,
        benc_belief_depth_enc_hidden_dim=args.benc_belief_depth_enc_hidden_dim,
        bdec_hidden_attn_hidden_dim=args.bdec_hidden_attn_hidden_dim,
        encoder_name=args.encoder_name,
    ).to(device)

    num_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_encoder_img = sum(p.numel() for p in model.encoder_img.parameters() if p.requires_grad)
    num_params_encoder_depth = sum(p.numel() for p in model.encoder_depth.parameters() if p.requires_grad)
    num_params_state_cell = num_params_total - num_params_encoder_img - num_params_encoder_depth
    logger.info(f"{num_params_total=}")
    logger.info(f"{num_params_encoder_img=}")
    logger.info(f"{num_params_encoder_depth=}")
    logger.info(f"{num_params_state_cell=}")

    if args.ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank] if args.use_cuda else None,
            output_device=args.local_rank if args.use_cuda else None,
        )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs // 1, gamma=0.5, verbose=False)

    trainer = Trainer(
        model=model,
        device=device,
        hidden_dim=hidden_dim,
        rnn_type=args.rnn_type,
        criterion_trans=criterion_trans,
        criterion_rot=criterion_rot,
        criterion_pose=criterion_pose,
        writer=writer,
    )

    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
        model.train()
        if args.ddp:
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
        if epoch % args.save_epoch_freq == 0:
            save_model(model, model_path)

        if epoch % args.val_epoch_freq == 0 and not args.do_overfit:
            model.eval()
            with torch.no_grad():
                val_stats = trainer.loader_forward(
                    val_loader,
                    stage="val",
                )
            print_stats(val_stats, logger, "val")
            for k, v in val_stats.items():
                history["val"][k].append(v)

            if args.use_es_val:
                early_stopping(loss=history["val"]["loss"][-1])

        if args.use_es_train:
            early_stopping(loss=history["train"]["loss"][-1])

        if early_stopping.do_stop:
            logger.warning(f"Early stopping on epoch {epoch}")
            break

    if args.do_debug:
        shutil.rmtree(logdir)
        return

    if is_main_process:
        save_model(model, model_path)

    logger.info(f"{logdir=}")
    logger.info(f"{logpath=}")

    if args.use_test_set and is_main_process:
        preds_dir = exp_tools["preds_dir"]
        preds_dir.mkdir(parents=True, exist_ok=True)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        test_dataset = VideoDataset(
            ds=YCBineoatDataset(**ycbi_kwargs),
            seq_len=args.seq_len_test,
            seq_step=1,
            seq_start=0,
            num_samples=1,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_stats = trainer.loader_forward(
            test_loader,
            save_preds=True,
            preds_dir=preds_dir,
            stage="test",
        )
        print_stats(test_stats, logger, "test")

        logger.info(f"saved to {preds_dir=} {preds_dir.name}")

    if not external_tools and is_main_process:
        exp.end()

    if args.ddp:
        dist.destroy_process_group()


def print_stats(train_stats, logger, stage):
    logger.info(f"## {stage.upper()} ##")
    LOSS_METRICS = ["loss", "loss_pose", "loss_depth"]
    ERROR_METRICS = ["r_err", "t_err"]
    ADDITIONAL_METRICS = ["add", "adds", "miou", "5deg5cm", "2deg2cm"]

    for stat_group in [LOSS_METRICS, ERROR_METRICS, ADDITIONAL_METRICS]:
        msg = []
        for k in stat_group:
            if k in train_stats:
                msg.append(f"{k}: {train_stats[k]:.4f}")
        logger.info(" | ".join(msg))


def save_results(batch_t, t_pred, rot_pred, preds_dir):
    # batch_t contains data for the t-th timestep in N sequences
    batch_size = len(batch_t["rgb"])
    for seq_idx in range(batch_size):
        rgb = batch_t["rgb"][seq_idx].cpu().numpy()
        name = Path(batch_t["rgb_path"][seq_idx]).stem
        pose = torch.eye(4)
        r_quat = rot_pred[seq_idx]
        pose[:3, :3] = quaternion_to_matrix(r_quat)
        pose[:3, 3] = t_pred[seq_idx] * 1e3
        pose = cast_to_numpy(pose)
        gt_pose = batch_t["pose"][seq_idx]
        gt_pose_formatted = convert_pose_quaternion_to_matrix(gt_pose)
        gt_pose_formatted[:3, 3] = gt_pose[:3].squeeze() * 1e3
        gt_pose_formatted = cast_to_numpy(gt_pose_formatted)
        seq_dir = preds_dir if batch_size == 1 else preds_dir / f"seq_{seq_idx}"
        pose_path = seq_dir / "poses" / f"{name}.txt"
        gt_path = seq_dir / "poses_gt" / f"{name}.txt"
        rgb_path = seq_dir / "rgb" / f"{name}.png"
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pose_path, "w") as f:
            for row in pose:
                f.write(" ".join(map(str, row)) + "\n")
        with open(gt_path, "w") as f:
            for row in gt_pose_formatted:
                f.write(" ".join(map(str, row)) + "\n")
        rgb = adjust_img_for_plt(rgb)
        rgb = rgb[..., ::-1]
        rgb_path = str(rgb_path)
        cv2.imwrite(rgb_path, rgb)


class Trainer:

    def __init__(
        self,
        model,
        device,
        hidden_dim,
        rnn_type,
        criterion_trans=None,
        criterion_rot=None,
        criterion_pose=None,
        writer=None,
    ):
        assert criterion_pose is not None or (
            criterion_rot is not None and criterion_trans is not None
        ), "Either pose or rot & trans criteria must be provided"

        self.model = model
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.criterion_trans = criterion_trans
        self.criterion_rot = criterion_rot
        self.criterion_pose = criterion_pose
        self.writer = writer

        self.use_pose_loss = criterion_pose is not None
        self.do_log = writer is not None
        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.epoch_counts_per_stage = defaultdict(int)

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        running_stats = defaultdict(float)
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_stats = self.batched_seq_forward(
                batched_seq=batched_seq,
                optimizer=optimizer,
                save_preds=save_preds,
                preds_dir=preds_dir,
                stage=stage,
            )

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                v = v.item() if isinstance(v, torch.Tensor) else v
                running_stats[k] += v
                if self.do_log:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            seq_pbar.set_postfix(
                {k: v / (seq_pack_idx + 1) for k, v in running_stats.items()},
            )

        for k, v in running_stats.items():
            running_stats[k] = v / len(loader)

        if self.do_log:
            for k, v in running_stats.items():
                self.writer.add_scalar(f"{stage}_epoch/{k}", v, self.epoch_counts_per_stage[stage])
        self.epoch_counts_per_stage[stage] += 1

        return running_stats

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):

        is_train = optimizer is not None

        batch_size = len(batched_seq[0]["rgb"])
        hx = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        cx = None if "gru" in self.rnn_type else torch.zeros(batch_size, self.hidden_dim).to(self.device)
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq))
        for t, batch_t in ts_pbar:
            if is_train:
                optimizer.zero_grad()
            rgb = batch_t["rgb"]
            seg_masks = batch_t["mask"]
            pose_gt = batch_t["pose"]
            depth = batch_t["depth"]

            outputs = self.model(rgb, depth, hx=hx, cx=cx)

            pts = batch_t["mesh_pts"]
            rot_pred, t_pred = outputs["rot"], outputs["t"]
            pose_pred = torch.stack(
                [convert_pose_quaternion_to_matrix(rt) for rt in torch.cat([rot_pred, t_pred], dim=1)]
            )
            pose_gt_mat = torch.stack(
                [convert_pose_quaternion_to_matrix(rt) for rt in torch.cat([pose_gt[:, 3:], pose_gt[:, :3]], dim=1)]
            )
            if self.use_pose_loss:
                loss_pose = self.criterion_pose(pose_pred, pose_gt_mat, pts)
                loss = loss_pose.clone()
            else:
                trans_labels = pose_gt[:, :3]
                rot_labels = pose_gt[:, 3:]
                loss_trans = self.criterion_trans(outputs["t"], trans_labels)
                loss_rot = self.criterion_rot(outputs["rot"], rot_labels)
                loss = loss_trans + loss_rot

            bbox_3d = batch_t["mesh_bbox"]
            diameter = batch_t["mesh_diameter"]
            m_batch = defaultdict(list)
            for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_pred, pose_gt_mat)):
                m_sample = calc_metrics(
                    pred_rt=pred_rt,
                    gt_rt=gt_rt,
                    pts=pts[sample_idx],
                    class_name=None,
                    use_miou=True,
                    bbox_3d=bbox_3d[sample_idx],
                    diameter=diameter[sample_idx],
                )
                for k, v in m_sample.items():
                    m_batch[k].append(v)
            m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
            seq_metrics = {k: seq_metrics[k] + v for k, v in m_batch_avg.items()}

            if self.do_log:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])
            self.ts_counts_per_stage[stage] += 1

            loss_depth = F.mse_loss(outputs["decoder_out"]["depth_final"], outputs["latent_depth"])
            loss += loss_depth
            # loss_priv = F.mse_loss(outputs["priv_decoded"], batch_t["priv"])
            # loss += loss_priv

            if is_train:
                loss.backward()
                optimizer.step()

            seq_stats["loss"] += loss
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose
            else:
                seq_stats["loss_rot"] += loss_rot
                seq_stats["loss_trans"] += loss_trans
            seq_stats["loss_depth"] += loss_depth

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, outputs["t"], outputs["rot"], preds_dir)

        for k, v in seq_stats.items():
            seq_stats[k] = v / len(batched_seq)
        for k, v in seq_metrics.items():
            seq_metrics[k] = v / len(batched_seq)

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
