import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from pose_tracking.models.direct_regr_cnn import DirectRegrCNN
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PROJ_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.bop import BOPDataset
from pose_tracking.losses import geodesic_loss
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.common import print_args
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

matplotlib.use("TKAgg")


def custom_collate_fn(batch):
    new_b = defaultdict(list)
    for k in batch[0].keys():
        new_b[k] = [d[k] for d in batch]
    for k, v in new_b.items():
        if isinstance(v[0], torch.Tensor):
            new_b[k] = torch.stack(v)
    return new_b


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, scene, transform=None):
        self.transform = transform
        self.num_samples = len(scene["frame_id"])
        self.scene = scene

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obj_idx = 0
        frame_id = self.scene["frame_id"][idx]
        rgb = self.scene["rgb"][idx]
        mask = self.scene["mask"][idx][obj_idx]
        rgb_path = self.scene["rgb_path"][idx]

        if self.transform:
            rgb = self.transform(rgb)
            mask = self.transform(mask)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        pose = self.scene["pose"][idx][obj_idx]
        r = pose[:3, :3]
        r_quat = matrix_to_quaternion(r)
        t = pose[:3, 3] / 1e3

        return {"frame_id": frame_id, "rgb": rgb, "mask": mask, "r": r_quat, "t": t, "rgb_path": rgb_path}

    def clone(self, idxs=None):
        scene = self.scene
        if idxs is not None:
            scene = {k: [v[i] for i in idxs] for k, v in scene.items()}
        return SceneDataset(scene, self.transform)


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

    if is_main_process:
        logger = prepare_logger(logpath=f"{logdir}/log.log", level="INFO")
        sys.excepthook = log_exception
    else:
        logger = prepare_logger(logpath=f"{logdir}/log.log", level="INFO")
        logger.remove()
    logger.info(f"PROJ_ROOT path is: {PROJ_DIR}")

    model = DirectRegrCNN().to(device)
    logger.info(f"model.parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    criterion_trans = nn.MSELoss()
    criterion_rot = geodesic_loss

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs // 1, gamma=0.5, verbose=False)

    transform = transforms.Compose([])

    early_stopping = EarlyStopping(patience=5, delta=0.1, verbose=True)

    split = "test"
    ds_name = args.ds_name
    ds_dir = DATA_DIR / ds_name
    seq_length = args.seq_length
    full_ds = BOPDataset(
        ds_dir,
        split,
        cad_dir=ds_dir / "models",
        seq_length=seq_length,
        use_keyframes=False,
        do_load_cad=False,
        include_depth=False,
        include_rgb=True,
        include_mask=True,
    )
    scene_idx = args.scene_idx
    scene = full_ds[scene_idx]
    scene_len = len(scene["frame_id"])
    logger.info(f"Scene length: {scene_len}")
    train_share = 0.8
    train_len = int(train_share * scene_len)
    scene_train = {k: v[:train_len] for k, v in scene.items()}
    scene_val = {k: v[train_len:] for k, v in scene.items()}

    train_dataset = SceneDataset(scene=scene_train, transform=transform)
    val_dataset = SceneDataset(scene=scene_val, transform=transform)

    if args.do_overfit:
        train_dataset = train_dataset.clone()
        val_dataset = train_dataset

    logger.info(f"{len(train_dataset)=}")

    collate_fn = custom_collate_fn
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

    for epoch in tqdm(range(1, args.num_epochs + 1), desc="Epochs"):
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        running_losses = defaultdict(float)
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            images = batch["rgb"]
            seg_masks = batch["mask"].float()
            trans_labels = batch["t"]
            rot_labels = batch["r"]
            images = images.to(device)
            seg_masks = seg_masks.to(device)
            trans_labels = trans_labels.to(device)
            rot_labels = rot_labels.to(device)

            optimizer.zero_grad()
            trans_output, rot_output = model(images, seg_masks)

            loss_trans = criterion_trans(trans_output, trans_labels)
            loss_rot = criterion_rot(rot_output, rot_labels)
            loss = loss_trans + loss_rot

            loss.backward()
            optimizer.step()

            running_losses["loss"] += loss
            running_losses["loss_rot"] += loss_rot
            running_losses["loss_trans"] += loss_trans

        logger.info(f"#### Epoch {epoch} ####")
        for k, v in running_losses.items():
            running_losses[k] = v / len(train_loader)

        for k, v in running_losses.items():
            running_loss = v.item()
            logger.info(f"{k}: {running_loss:.4f}")
            history["train"][k].append(running_loss)

        lr_scheduler.step()
        if epoch % args.save_epoch_freq == 0:
            save_model(model, model_path)

        if epoch % args.val_epoch_freq == 0 and not args.do_overfit:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["rgb"]
                    seg_masks = batch["mask"]
                    trans_labels = batch["t"]
                    rot_labels = batch["r"]
                    images = images.to(device)
                    seg_masks = seg_masks.to(device)
                    trans_labels = trans_labels.to(device)
                    rot_labels = rot_labels.to(device)

                    trans_output, rot_output = model(images, seg_masks)

                    loss_trans = criterion_trans(trans_output, trans_labels)
                    loss_rot = criterion_rot(rot_output, rot_labels)
                    loss = loss_trans + loss_rot

                    val_loss += loss.item()
            val_loss /= len(val_loader)
            logger.info(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

            if args.use_early_stopping:
                early_stopping(loss=val_loss)

        if args.do_overfit and args.use_early_stopping:
            early_stopping(loss=running_losses["loss"])

        if early_stopping.do_stop:
            logger.warning(f"Early stopping on epoch {epoch}")
            break

    if args.do_debug:
        shutil.rmtree(logdir)
        return

    save_model(model, model_path)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    test_loader = train_loader if args.do_overfit else val_loader
    for batch in tqdm(test_loader):
        images = batch["rgb"]
        seg_masks = batch["mask"]
        rgb_paths = batch["rgb_path"]
        images = images.to(device)
        seg_masks = seg_masks.to(device).float()

        trans_output, rot_output = model(images, seg_masks)
        for i in range(len(images)):
            rgb = images[i].cpu().numpy()
            name = Path(rgb_paths[i]).stem
            pose = torch.eye(4)
            r_quat = rot_output[i]
            pose[:3, :3] = quaternion_to_matrix(r_quat)
            pose[:3, 3] = trans_output[i] * 1e3
            gt_pose = torch.eye(4)
            gt_pose[:3, :3] = quaternion_to_matrix(batch["r"][i].squeeze())
            gt_pose[:3, 3] = batch["t"][i].squeeze() * 1e3
            pose = pose.cpu().numpy()
            gt_pose = gt_pose.cpu().numpy()
            pose_path = preds_dir / "poses" / f"{name}.txt"
            gt_path = preds_dir / "poses_gt" / f"{name}.txt"
            rgb_path = preds_dir / "rgb" / f"{name}.png"
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pose_path, "w") as f:
                for row in pose:
                    f.write(" ".join(map(str, row)) + "\n")
            with open(gt_path, "w") as f:
                for row in gt_pose:
                    f.write(" ".join(map(str, row)) + "\n")
            rgb = (rgb * 255).astype(np.uint8)
            rgb = rgb.transpose(1, 2, 0)
            rgb_path = str(rgb_path)
            cv2.imwrite(rgb_path, rgb)

    logger.info(f"saved to {preds_dir=} {preds_dir.name}")

    if args.ddp:
        dist.destroy_process_group()


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
