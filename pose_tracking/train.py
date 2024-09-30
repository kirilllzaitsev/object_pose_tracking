import argparse
import os
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import DATA_DIR, PROJ_DIR, WORKSPACE_DIR, logger
from pose_tracking.dataset.bop import BOPDataset
from pose_tracking.losses import geodesic_loss
from pose_tracking.utils.common import print_args
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

matplotlib.use("TKAgg")


def parse_args():
    parser = argparse.ArgumentParser()

    train_args = parser.add_argument_group("Training arguments")
    train_args.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_args.add_argument("--validate_every", type=int, default=5, help="Validate every N epochs")
    train_args.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    train_args.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel")
    train_args.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    train_args.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    train_args.add_argument("--seed", type=int, default=10, help="Random seed")
    train_args.add_argument("--use_early_stopping", action="store_true", help="Use early stopping")
    train_args.add_argument("--do_overfit", action="store_true", help="Overfit setting")
    train_args.add_argument("--do_debug", action="store_true", help="Debugging setting")

    return parser.parse_args()


def postprocess_args(args):
    if args.do_overfit:
        args.dropout_prob = 0.0

    return args


class PoseCNN(nn.Module):
    def __init__(self, dropout_prob=0.3, backbone_name="mobilenet_v3_large"):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=True)
            last_backbone_ch = 2048
        elif backbone_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(pretrained=True)
            last_backbone_ch = 960
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        self.backbone_name = backbone_name

        self.do_modify_first_conv = True
        self.do_modify_first_conv = False
        if self.do_modify_first_conv:
            self.modify_first_conv(backbone)

        modules = list(backbone.children())[:-2]
        self.features = nn.Sequential(*modules)

        self.translation_head = nn.Sequential(
            nn.Conv2d(last_backbone_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 3),
        )

        self.rotation_head = nn.Sequential(
            nn.Conv2d(last_backbone_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def modify_first_conv(self, backbone):
        if self.backbone_name == "mobilenet_v3_large":
            original_conv = backbone.features[0][0]
        else:
            original_conv = backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            new_conv.weight[:, 3:, :, :] = original_conv.weight.mean(dim=1, keepdim=True)

        if self.backbone_name == "mobilenet_v3_large":
            backbone.features[0][0] = new_conv
        else:
            backbone.conv1 = new_conv

    def forward(self, x, segm_mask):
        if self.do_modify_first_conv:
            x = torch.cat([x, segm_mask], dim=1)
        features = self.features(x)

        trans_output = self.translation_head(features)

        rot_output = self.rotation_head(features)

        rot_output = rot_output / rot_output.norm(dim=1, keepdim=True)

        return trans_output, rot_output


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

        # rgb = (rgb * mask).float()

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
    args = postprocess_args(args)
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

    model = PoseCNN().to(device)
    logger.info(f"model.parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    criterion_trans = nn.MSELoss()
    criterion_rot = geodesic_loss

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 1, gamma=0.5, verbose=False)

    transform = transforms.Compose([])

    early_stopping = EarlyStopping(patience=5, delta=0.1, verbose=True)

    split = "test"
    ds_name = "ycbv"
    ds_name = "lm"
    ds_dir = DATA_DIR / ds_name
    full_ds = BOPDataset(
        ds_dir,
        split,
        cad_dir=ds_dir / "models",
        seq_length=None,
        use_keyframes=False,
        do_load_cad=False,
        include_depth=False,
        include_rgb=True,
        include_masks=True,
    )
    scene = full_ds[0]
    scene_len = len(scene["frame_id"])
    logger.info(f"Scene length: {scene_len}")
    train_share = 0.8
    train_len = int(train_share * scene_len)
    scene_train = {k: v[:train_len] for k, v in scene.items()}
    scene_val = {k: v[train_len:] for k, v in scene.items()}

    train_dataset = SceneDataset(scene=scene_train, transform=transform)
    val_dataset = SceneDataset(scene=scene_val, transform=transform)

    if args.do_overfit:
        train_dataset = train_dataset.clone([0])
        val_dataset = train_dataset

    logger.info(f"{len(train_dataset)=}")

    preds_base_dir = f"{WORKSPACE_DIR}/preds"
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    preds_dir = Path(preds_base_dir) / now
    preds_dir.mkdir(parents=True, exist_ok=True)
    model_path = preds_dir / "model.pth"

    collate_fn = None
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

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
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
            v /= len(train_loader)
            running_loss = v.item()
            logger.info(f"{k}: {running_loss:.4f}")
            history["train"][k].append(running_loss)

        lr_scheduler.step()
        if epoch % args.save_every == 0:
            save_model(model, model_path)

        if epoch % args.validate_every == 0 and not args.do_overfit:
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

                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader)
            logger.info(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

            if args.use_early_stopping:
                early_stopping(loss=val_loss)
                if early_stopping.do_stop:
                    logger.warning(f"Early stopping on epoch {epoch}")
                    break

    if args.do_debug:
        shutil.rmtree(preds_dir)
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
            name = rgb_paths[i].split("/")[-1]
            pose = torch.eye(4)
            r_quat = rot_output[i]
            pose[:3, :3] = quaternion_to_matrix(r_quat)
            pose[:3, 3] = trans_output[i] * 1e3
            pose = pose.cpu().numpy()
            pose_path = preds_dir / "poses" / f"{name}.txt"
            rgb_path = preds_dir / "rgb" / name
            pose_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pose_path, "w") as f:
                for row in pose:
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
