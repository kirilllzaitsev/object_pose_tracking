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
import torch.nn.functional as F
import torch.optim as optim
from pose_tracking.callbacks import EarlyStopping
from pose_tracking.config import (
    DATA_DIR,
    PROJ_DIR,
    YCB_MESHES_DIR,
    log_exception,
    prepare_logger,
)
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.dataset.ds_common import batch_seq_collate_fn, seq_collate_fn
from pose_tracking.dataset.transforms import get_transforms
from pose_tracking.dataset.video_ds import VideoDataset
from pose_tracking.config import default_logger
from pose_tracking.losses import compute_chamfer_dist, kpt_cross_ratio_loss
from pose_tracking.metrics import calc_metrics
from pose_tracking.models.encoders import is_param_part_of_encoders
from pose_tracking.utils.args_parsing import parse_args
from pose_tracking.utils.common import print_args
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    cam_to_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
    rot_mat_from_6d,
    rotate_pts_batch,
)
from pose_tracking.utils.misc import set_seed
from pose_tracking.utils.pipe_utils import (
    Printer,
    create_tools,
    get_full_ds,
    get_model,
    get_obj_ds,
    get_trainer,
    log_artifacts,
    log_exp_meta,
    log_model_meta,
    print_stats,
    reduce_metric,
    save_results,
)
from pose_tracking.utils.pose import convert_pose_quaternion_to_matrix
from pose_tracking.utils.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
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

    collate_fn = seq_collate_fn if args.model_name == "cnnlstm" else batch_seq_collate_fn
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

    history = defaultdict(lambda: defaultdict(list))

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
        do_debug=False,
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        use_rnn=True,
        use_obs_belief=True,
        world_size=1,
        do_log_every_ts=False,
        do_log_every_seq=True,
        use_ddp=False,
        use_prev_pose_condition=False,
        do_predict_rel_pose=False,
        do_predict_kpts=False,
        logger=None,
        vis_epoch_freq=None,
        do_vis=False,
        exp_dir=None,
    ):
        assert criterion_pose is not None or (
            criterion_rot is not None and criterion_trans is not None
        ), "Either pose or rot & trans criteria must be provided"

        self.do_debug = do_debug
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.use_rnn = use_rnn
        self.use_obs_belief = use_obs_belief
        self.world_size = world_size
        self.logger = default_logger if logger is None else logger
        self.vis_epoch_freq = vis_epoch_freq
        self.do_log_every_ts = do_log_every_ts
        self.do_log_every_seq = do_log_every_seq
        self.use_ddp = use_ddp
        self.use_prev_pose_condition = use_prev_pose_condition
        self.do_predict_rel_pose = do_predict_rel_pose
        self.do_predict_kpts = do_predict_kpts
        self.do_vis = do_vis
        self.exp_dir = exp_dir

        self.use_pose_loss = criterion_pose is not None
        self.do_log = writer is not None
        self.use_optim_every_ts = not use_rnn

        self.model = model
        self.device = device
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.criterion_trans = criterion_trans
        self.criterion_rot = criterion_rot
        self.criterion_pose = criterion_pose
        self.writer = writer
        self.processed_data = defaultdict(list)

        self.seq_counts_per_stage = defaultdict(int)
        self.ts_counts_per_stage = defaultdict(int)
        self.train_epoch_count = 0
        self.vis_dir = f"{self.exp_dir}/vis"

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        seq_pbar = tqdm(loader, desc="Seq", leave=False)
        do_vis = self.do_vis and self.train_epoch_count % self.vis_epoch_freq == 0

        for seq_pack_idx, batched_seq in enumerate(seq_pbar):
            seq_stats = self.batched_seq_forward(
                batched_seq=batched_seq,
                optimizer=optimizer,
                save_preds=save_preds,
                preds_dir=preds_dir,
                stage=stage,
                do_vis=do_vis,
            )

            for k, v in {**seq_stats["losses"], **seq_stats["metrics"]}.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                    if self.use_ddp:
                        reduce_metric(v, world_size=self.world_size)
                running_stats[k] += v
                if self.do_log:
                    self.writer.add_scalar(f"{stage}_seq/{k}", v, self.seq_counts_per_stage[stage])
            self.seq_counts_per_stage[stage] += 1

            seq_pbar.set_postfix(
                {k: v / (seq_pack_idx + 1) for k, v in running_stats.items()},
            )

            do_vis = False  # only do vis for the first seq

        for k, v in running_stats.items():
            running_stats[k] = v / len(loader)

        if self.do_log and self.do_log_every_seq:
            for k, v in running_stats.items():
                self.writer.add_scalar(f"{stage}_epoch/{k}", v, self.train_epoch_count)

        return running_stats

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
    ):

        is_train = optimizer is not None

        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(float)
        seq_metrics = defaultdict(float)
        ts_pbar = tqdm(enumerate(batched_seq), desc="Timestep", leave=False, total=len(batched_seq))

        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts
        if do_opt_in_the_end:
            optimizer.zero_grad()
            total_loss = 0

        if self.use_ddp:
            self.model.module.reset_state(batch_size, device=self.device)
        else:
            self.model.reset_state(batch_size, device=self.device)
        if self.do_debug:
            self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})

        if do_vis:
            vis_batch_idxs = np.random.choice(batch_size, 2, replace=False)
            vis_data = defaultdict(list)

        abs_prev_pose = None  # the processed ouput of the model that matches model's expected format
        prev_model_out = None  # the raw ouput of the model
        prev_gt_mat = None  # prev gt in matrix form

        for t, batch_t in ts_pbar:
            if do_opt_every_ts:
                optimizer.zero_grad()
            rgb = batch_t["rgb"]
            seg_masks = batch_t["mask"]
            pose_gt = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]

            if abs_prev_pose is None and self.do_predict_rel_pose:
                assert (
                    not self.do_predict_6d_rot and not self.do_predict_2d_t
                ), "Relative pose prediction is not supported with 6d rot or 2d t"
                abs_prev_pose = {"t": pose_gt[:, :3], "rot": pose_gt[:, 3:]}

            outputs = self.model(rgb, depth, prev_pose=abs_prev_pose if self.do_predict_rel_pose else prev_model_out)
            prev_model_out = {"t": outputs["t"], "rot": outputs["rot"]}

            rot_pred, t_pred = outputs["rot"], outputs["t"]

            if self.do_predict_6d_rot:
                rot_pred = rot_mat_from_6d(rot_pred)

            img_size = rgb.shape[-2:]
            h, w = img_size
            t_gt = pose_gt[:, :3]
            rot_gt = pose_gt[:, 3:]
            intrinsics = batch_t["intrinsics"]

            if self.do_predict_2d_t:
                # 3d t_pred will be used only for metrics
                t_pred_2d_denorm = t_pred.detach().clone()
                t_pred_2d_denorm[:, 0] = t_pred_2d_denorm[:, 0] * w
                t_pred_2d_denorm[:, 1] = t_pred_2d_denorm[:, 1] * h

                depth_gt = t_gt[:, 2]
                center_depth_pred = outputs["center_depth"]
                t_pred_2d_backproj = []
                for sample_idx in range(len(depth_gt)):
                    t_pred_2d_backproj.append(
                        backproj_2d_to_3d(
                            t_pred_2d_denorm[sample_idx][None], center_depth_pred[sample_idx], intrinsics[sample_idx]
                        ).squeeze()
                    )
                t_pred = torch.stack(t_pred_2d_backproj).to(rot_pred.device)

            pose_gt_mat = torch.stack([convert_pose_quaternion_to_matrix(rt) for rt in pose_gt])
            prev_gt_mat = pose_gt_mat
            if self.do_predict_6d_rot:
                pose_pred = torch.eye(4).repeat(batch_size, 1, 1).to(self.device)
                pose_pred[:, :3, :3] = rot_pred
                pose_pred[:, :3, 3] = t_pred
            else:
                pose_pred = torch.stack(
                    [convert_pose_quaternion_to_matrix(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
                )

            if self.do_predict_rel_pose:
                prev_pose_mat = torch.stack(
                    [
                        convert_pose_quaternion_to_matrix(rt)
                        for rt in torch.cat([abs_prev_pose["t"], abs_prev_pose["rot"]], dim=1)
                    ]
                )
                pose_pred = egocentric_delta_pose_to_pose(
                    prev_pose_mat,
                    trans_delta=pose_pred[:, :3, 3],
                    rot_mat_delta=pose_pred[:, :3, :3],
                    do_couple_rot_t=False,
                )

            # -- pose_pred is abs, t_pred/rot_pred can be rel or abs

            if self.use_pose_loss:
                loss_pose = self.criterion_pose(pose_pred, pose_gt_mat, pts)
                loss = loss_pose.clone()
            else:
                if self.do_predict_rel_pose:
                    t_gt_rel, rot_gt_rel = pose_to_egocentric_delta_pose(pose_gt_mat, prev_gt_mat)
                    rot_gt_rel = matrix_to_quaternion(rot_gt_rel)
                if self.do_predict_2d_t:
                    t_gt_2d = cam_to_2d(t_gt.unsqueeze(1), intrinsics).squeeze(1)
                    t_gt_2d_norm = t_gt_2d.clone()
                    t_gt_2d_norm[:, 0] = t_gt_2d_norm[:, 0] / w
                    t_gt_2d_norm[:, 1] = t_gt_2d_norm[:, 1] / h

                    t_pred_2d = outputs["t"]
                    loss_t_2d = torch.abs(t_pred_2d - t_gt_2d_norm).mean()
                    loss_center_depth = torch.abs(center_depth_pred - depth_gt).mean()

                    loss_t = loss_t_2d + loss_center_depth
                else:
                    if self.do_predict_rel_pose:
                        loss_t = self.criterion_trans(t_pred, t_gt_rel)
                    else:
                        loss_t = self.criterion_trans(t_pred, t_gt)
                if self.do_predict_6d_rot:
                    loss_rot = torch.abs(
                        rotate_pts_batch(pose_pred[:, :3, :3], pts) - rotate_pts_batch(pose_gt_mat[:, :3, :3], pts)
                    ).mean()
                else:
                    if self.do_predict_rel_pose:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt_rel)
                    else:
                        loss_rot = self.criterion_rot(rot_pred, rot_gt)
                loss = loss_rot + loss_t

            bbox_3d = batch_t["mesh_bbox"]
            diameter = batch_t["mesh_diameter"]
            m_batch = defaultdict(list)
            nan_count = 0
            for sample_idx, (pred_rt, gt_rt) in enumerate(zip(pose_pred, pose_gt_mat)):
                m_sample = calc_metrics(
                    pred_rt=pred_rt,
                    gt_rt=gt_rt,
                    pts=pts[sample_idx],
                    class_name=None,
                    use_miou=True,
                    bbox_3d=bbox_3d[sample_idx],
                    diameter=diameter[sample_idx],
                    is_meters=True,
                    log_fn=print if self.logger is None else self.logger.warning,
                )
                for k, v in m_sample.items():
                    m_batch[k].append(v)
                if any(np.isnan(v) for v in m_sample.values()):
                    nan_count += 1

            m_batch_avg = {k: np.mean(v) for k, v in m_batch.items()}
            for k, v in m_batch_avg.items():
                seq_metrics[k] += v
            seq_metrics["nan_count"] += nan_count

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])
            self.ts_counts_per_stage[stage] += 1

            if self.use_obs_belief:
                loss_depth = F.mse_loss(outputs["decoder_out"]["depth_final"], outputs["latent_depth"])
            else:
                loss_depth = torch.tensor(0.0).to(self.device)
            loss += loss_depth

            if "priv_decoded" in outputs:
                loss_priv = compute_chamfer_dist(outputs["priv_decoded"], batch_t["priv"])
                loss += loss_priv * 0.01

            if self.do_predict_kpts:
                kpts_pred = outputs["kpts"]
                kpts_gt = batch_t["bbox_2d_kpts"].float()
                loss_kpts = F.huber_loss(kpts_pred, kpts_gt)
                bbox_2d_kpts_collinear_idxs = batch_t["bbox_2d_kpts_collinear_idxs"]
                loss_cr = kpt_cross_ratio_loss(kpts_pred, bbox_2d_kpts_collinear_idxs)
                loss += loss_kpts
                loss += loss_cr * 0.01

            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                optimizer.step()
            elif do_opt_in_the_end:
                total_loss += loss

            seq_stats["loss"] += loss
            seq_stats["loss_depth"] += loss_depth
            if self.use_pose_loss:
                seq_stats["loss_pose"] += loss_pose
            else:
                seq_stats["loss_rot"] += loss_rot
                seq_stats["loss_t"] += loss_t
            if "priv_decoded" in outputs:
                seq_stats["loss_priv"] += loss_priv
            if self.do_predict_kpts:
                seq_stats["loss_kpts"] += loss_kpts
                seq_stats["loss_cr"] += loss_cr

            if self.do_predict_rel_pose:
                abs_rot_quat = matrix_to_quaternion(pose_pred[:, :3, :3])
                abs_prev_pose = {"t": pose_pred[:, :3, 3], "rot": abs_rot_quat}
            else:
                abs_prev_pose = {"t": t_pred, "rot": rot_pred}
                if self.do_predict_2d_t:
                    abs_prev_pose["center_depth"] = center_depth_pred
            abs_prev_pose = {k: v.detach() for k, v in abs_prev_pose.items()}

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, t_pred, rot_pred, preds_dir)

            if do_vis:
                # save inputs to the exp dir
                for k in ["rgb", "mask", "depth", "intrinsics", "mesh_bbox"]:
                    vis_data[k].append([batch_t[k][i] for i in vis_batch_idxs])
                vis_data["pose_pred"].append(pose_pred[vis_batch_idxs].detach())
                vis_data["pose_gt_mat"].append(pose_gt_mat[vis_batch_idxs])

            if self.do_debug:
                # add everything to processed_data
                self.processed_data["state"].append({"hx": self.model.hx, "cx": self.model.cx})
                self.processed_data["rgb"].append(rgb)
                self.processed_data["seg_masks"].append(seg_masks)
                self.processed_data["pose_gt"].append(pose_gt)
                self.processed_data["pose_gt_mat"].append(pose_gt_mat)
                self.processed_data["depth"].append(depth)
                self.processed_data["rot_pred"].append(rot_pred)
                self.processed_data["t_pred"].append(t_pred)
                if self.do_predict_2d_t:
                    self.processed_data["t_gt_2d_norm"].append(t_gt_2d_norm)
                if "priv_decoded" in outputs:
                    self.processed_data["priv_decoded"].append(outputs["priv_decoded"])
                self.processed_data["pose_pred"].append(pose_pred)
                self.processed_data["abs_prev_pose"].append(abs_prev_pose)
                self.processed_data["pts"].append(pts)
                self.processed_data["bbox_3d"].append(bbox_3d)
                self.processed_data["diameter"].append(diameter)
                self.processed_data["loss"].append(loss)
                self.processed_data["m_batch"].append(m_batch)
                self.processed_data["loss_depth"].append(loss_depth)
                self.processed_data["prev_model_out"].append(prev_model_out)
                if self.use_pose_loss:
                    self.processed_data["loss_pose"].append(loss_pose)
                else:
                    self.processed_data["loss_rot"].append(loss_rot)
                    self.processed_data["loss_t"].append(loss_t)

        if do_opt_in_the_end:
            total_loss /= len(batched_seq)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()

        for k, v in seq_stats.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            seq_stats[k] = v / len(batched_seq)
        for k, v in seq_metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            seq_metrics[k] = v / len(batched_seq)
        if seq_metrics["nan_count"] == 0:
            seq_metrics.pop("nan_count")

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = f"{self.vis_dir}/epoch_{self.train_epoch_count}.pt"
            torch.save(vis_data, save_vis_path)
            self.logger.info(f"Saved vis data to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
        }


class TrainerVideopose(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loader_forward(
        self,
        loader,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
    ):
        if stage == "train":
            self.train_epoch_count += 1
        running_stats = defaultdict(float)
        return running_stats


if __name__ == "__main__":
    main()
