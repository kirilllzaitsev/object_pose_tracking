import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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
    ARTIFACTS_DIR,
    DATA_DIR,
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
from pose_tracking.losses import geodesic_loss
from pose_tracking.models.cnnlstm import RecurrentCNN
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.common import adjust_img_for_plt, print_args
from pose_tracking.utils.misc import set_seed, to_numpy
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.rotation_conversions import quaternion_to_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

matplotlib.use("TKAgg")


def main():
    args = parse_args()
    print_args(args)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ddp:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(np.random.randint(20000, 30000))

        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        is_main_process = rank == 0
    else:
        is_main_process = True

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logdir = f"{ARTIFACTS_DIR}/{args.exp_name}_{now}"
    writer = SummaryWriter(log_dir=logdir)
    preds_base_dir = f"{logdir}/preds"
    preds_dir = Path(preds_base_dir) / now
    preds_dir.mkdir(parents=True, exist_ok=True)
    model_path = preds_dir / "model.pth"

    logger = prepare_logger(logpath=f"{logdir}/log.log", level="INFO")
    if is_main_process:
        sys.excepthook = log_exception
    else:
        logger.remove()
    logger.info(f"PROJ_ROOT path is: {PROJ_DIR}")

    criterion_trans = nn.MSELoss()
    criterion_rot = geodesic_loss

    transform = get_transforms()

    early_stopping = EarlyStopping(patience=args.es_patience, delta=args.es_delta, verbose=True)

    split = "test"
    ds_name = args.ds_name
    ds_dir = DATA_DIR / ds_name
    seq_length = args.seq_length
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
        seq_len=args.seq_length,
        seq_step=args.seq_step,
        seq_start=args.seq_start,
        num_samples=args.num_samples,
    )
    full_ds = video_ds
    scene_len = len(full_ds)
    logger.info(f"Scene length: {scene_len}")
    train_share = 0.8
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
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs // 1, gamma=0.5, verbose=False)

    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        train_losses = loader_forward(
            train_loader,
            model,
            device,
            criterion_trans=criterion_trans,
            criterion_rot=criterion_rot,
            optimizer=optimizer,
        )

        logger.info(f"# Epoch {epoch} #")
        logger.info("## TRAIN ##")
        for k, v in train_losses.items():
            running_loss = v.item()
            logger.info(f"{k}: {running_loss:.4f}")
            history["train"][k].append(running_loss)

        lr_scheduler.step()
        if epoch % args.save_epoch_freq == 0:
            save_model(model, model_path)

        if epoch % args.val_epoch_freq == 0 and not args.do_overfit:
            model.eval()
            with torch.no_grad():
                val_losses = loader_forward(
                    val_loader,
                    model,
                    device,
                    criterion_trans=criterion_trans,
                    criterion_rot=criterion_rot,
                )
            logger.info("## VAL ##")
            for k, v in val_losses.items():
                running_loss = v.item()
                logger.info(f"{k}: {running_loss:.4f}")
                history["val"][k].append(running_loss)

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

    save_model(model, model_path)

    if args.use_test_set and is_main_process:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        test_dataset = VideoDataset(
            ds=YCBineoatDataset(**ycbi_kwargs),
            seq_len=100,
            # seq_len=None,
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
        loader_forward(
            test_loader,
            model,
            device,
            criterion_trans=criterion_trans,
            criterion_rot=criterion_rot,
            save_preds=True,
            preds_dir=preds_dir,
        )

        logger.info(f"saved to {preds_dir=} {preds_dir.name}")

    if args.ddp:
        dist.destroy_process_group()


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
        pose = to_numpy(pose)
        gt_pose = batch_t["pose"][seq_idx]
        gt_pose_formatted = convert_pose_quaternion_to_matrix(gt_pose)
        gt_pose_formatted[:3, 3] = gt_pose[:3].squeeze() * 1e3
        gt_pose_formatted = to_numpy(gt_pose_formatted)
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


def loader_forward(
    train_loader, model, device, criterion_trans, criterion_rot, optimizer=None, save_preds=False, preds_dir=None
):
    running_losses = defaultdict(float)
    seq_pbar = tqdm(train_loader, desc="Seq", leave=False)
    for seq_pack_idx, batched_seq in enumerate(seq_pbar):
        seq_losses = batched_seq_forward(
            batched_seq=batched_seq,
            model=model,
            device=device,
            criterion_trans=criterion_trans,
            criterion_rot=criterion_rot,
            optimizer=optimizer,
            save_preds=save_preds,
            preds_dir=preds_dir,
        )

        for k, v in seq_losses.items():
            running_losses[k] += v

        seq_pbar.set_postfix(
            {k: v / (seq_pack_idx + 1) for k, v in running_losses.items()},
        )

    for k, v in running_losses.items():
        running_losses[k] = v / len(train_loader)
    return running_losses


def batched_seq_forward(
    batched_seq, model, device, criterion_trans, criterion_rot, optimizer=None, save_preds=False, preds_dir=None
):
    batched_seq = transfer_batch_to_device(batched_seq, device)
    batch_size = len(batched_seq[0]["rgb"])
    hidden_dim = model.hidden_dim
    hx = torch.zeros(batch_size, hidden_dim).to(device)
    cx = None if "gru" in model.rnn_type else torch.zeros(batch_size, hidden_dim).to(device)
    ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False)
    is_train = optimizer is not None
    seq_losses = defaultdict(float)
    for t, batch_t in ts_pbar:
        if is_train:
            optimizer.zero_grad()
        rgb = batch_t["rgb"]
        seg_masks = batch_t["mask"]
        pose_gt = batch_t["pose"]
        depth = batch_t["depth"]

        outputs = model(rgb, depth, hx=hx, cx=cx)

        trans_labels = pose_gt[:, :3]
        rot_labels = pose_gt[:, 3:]
        loss_trans = criterion_trans(outputs["t"], trans_labels)
        loss_rot = criterion_rot(outputs["rot"], rot_labels)
        loss = loss_trans + loss_rot

        loss_depth = F.mse_loss(outputs["decoder_out"]["depth_final"], outputs["latent_depth"])
        loss += loss_depth
        # loss_priv = F.mse_loss(outputs["priv_decoded"], batch_t["priv"])
        # loss += loss_priv

        if is_train:
            loss.backward()
            optimizer.step()

        seq_losses["loss"] += loss
        seq_losses["loss_rot"] += loss_rot
        seq_losses["loss_trans"] += loss_trans
        seq_losses["loss_depth"] += loss_depth

        if save_preds:
            assert preds_dir is not None, "preds_dir must be provided for saving predictions"
            save_results(batch_t, outputs["t"], outputs["rot"], preds_dir)

        ts_pbar.set_postfix({k: v / (t + 1) for k, v in seq_losses.items()})

    return seq_losses


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
