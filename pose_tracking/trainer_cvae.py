import functools
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
from pose_tracking.dataset.dataloading import transfer_batch_to_device
from pose_tracking.losses import compute_chamfer_dist
from pose_tracking.trainer import Trainer
from pose_tracking.utils.artifact_utils import save_results
from pose_tracking.utils.common import detach_and_cpu
from pose_tracking.utils.geom import (
    backproj_2d_to_3d,
    convert_2d_t_to_3d,
    convert_3d_t_for_2d,
    egocentric_delta_pose_to_pose,
    pose_to_egocentric_delta_pose,
)
from pose_tracking.utils.kpt_utils import (
    get_pose_from_3d_2d_matches,
)
from pose_tracking.utils.pose import convert_r_t_to_rt
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class TrainerCVAE(Trainer):

    def __init__(
        self,
        *args,
        kl_loss_coef=1,
        **kwargs,
    ):
        self.kl_loss_coef = kl_loss_coef
        super().__init__(*args, **kwargs)

    def batched_seq_forward(
        self,
        batched_seq,
        *,
        optimizer=None,
        save_preds=False,
        preds_dir=None,
        stage="train",
        do_vis=False,
        last_step_state=None,
    ):

        is_train = stage == "train"
        do_opt_every_ts = is_train and self.use_optim_every_ts
        do_opt_in_the_end = is_train and not self.use_optim_every_ts

        seq_length = len(batched_seq)
        batch_size = len(batched_seq[0]["rgb"])
        batched_seq = transfer_batch_to_device(batched_seq, self.device)

        seq_stats = defaultdict(list)
        seq_metrics = defaultdict(list)
        ts_pbar = tqdm(
            enumerate(batched_seq),
            desc="Timestep",
            leave=False,
            total=len(batched_seq),
            disable=True,
        )

        total_losses = []

        if self.do_debug:
            self.processed_data["state"].append(detach_and_cpu({"hx": self.model.hx, "cx": self.model.cx}))

        if do_vis:
            vis_batch_idxs = list(range(min(batch_size, 8)))
            vis_data = defaultdict(list)

        pose_prev_pred_abs = None
        out_prev = None  # raw ouput of the model at prev step
        pose_mat_prev_gt_abs = None
        prev_latent = None
        prev_pose = None
        state = None
        do_skip_first_step = False
        prev_kpts = None

        for t, batch_t in ts_pbar:
            rgb = batch_t["rgb"]
            mask = batch_t["mask"]
            pose_gt_abs = batch_t["pose"]
            depth = batch_t["depth"]
            pts = batch_t["mesh_pts"]
            intrinsics = batch_t["intrinsics"]
            bbox_2d = batch_t["bbox_2d"]
            features_rgb = batch_t.get("features_rgb")
            h, w = rgb.shape[-2:]
            hw = (h, w)
            t_gt_abs = pose_gt_abs[:, :3]
            rot_gt_abs = pose_gt_abs[:, 3:]

            if self.do_predict_rel_pose:
                if t == 0 and last_step_state is not None:
                    pose_mat_prev_gt_abs = last_step_state["pose_mat_prev_gt_abs"]
                    pose_prev_pred_abs = last_step_state["pose_prev_pred_abs"]
                    prev_latent = last_step_state.get("prev_latent")
                    state = last_step_state.get("state")
                elif t == 0:
                    do_skip_first_step = True
                    if self.do_perturb_init_gt_for_rel_pose and is_train:
                        noise_t = (
                            torch.rand_like(t_gt_abs)
                            * 0.02
                            * (torch.randint(0, 2, t_gt_abs.shape) * 2 - 1).to(self.device)
                        )
                        noise_rot_mat = torch.stack([torch.eye(3) for _ in range(batch_size)])
                        for i in range(batch_size):
                            j = torch.randint(0, 3, (1,))
                            angle = np.random.uniform(1) * 5
                            if j == 0:
                                angles = [angle, 0, 0]
                            elif j == 1:
                                angles = [0, angle, 0]
                            else:
                                angles = [0, 0, angle]

                            noise_rot_mat[i] = torch.tensor(
                                Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                            )
                        noise_rot_mat = noise_rot_mat.to(self.device)
                        rot_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])[
                            ..., :3, :3
                        ]
                        rot_mat_gt_abs = torch.bmm(rot_mat_gt_abs, noise_rot_mat)
                        rot_gt_abs = self.rot_mat_to_vector_converter_fn(rot_mat_gt_abs)
                    else:
                        noise_t = 0

                    pose_mat_prev_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])

                    out = self.model(
                        rgb,
                        pose=None,
                        depth=depth,
                        bbox=bbox_2d,
                        state=None,
                        features_rgb=features_rgb,
                    )
                    state = out["state"]

                    prev_latent = out["latent"] if self.use_prev_latent else None

                    if self.do_predict_abs_pose:
                        t_abs_pose, rot_abs_pose = out["t_abs_pose"], out["rot_abs_pose"]
                        losses_abs_pose = self.calc_abs_pose_loss_for_rel(
                            pose_gt_abs, t_pred_abs_pose=t_abs_pose, rot_pred_abs_pose=rot_abs_pose
                        )
                        total_losses.append(losses_abs_pose["loss_abs_pose"])

                        pose_prev_pred_abs = {"t": t_abs_pose, "rot": rot_abs_pose}

                        for k, v in losses_abs_pose.items():
                            seq_stats[k].append(v.item())
                    else:
                        pose_prev_pred_abs = {"t": t_gt_abs + noise_t, "rot": rot_gt_abs}

                    if self.use_pnp_for_rot_pred:
                        prev_kpts = out["kpts"]

                    if save_preds:
                        assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                        save_results(batch_t, pose_mat_prev_gt_abs, preds_dir, gt_pose=pose_mat_prev_gt_abs)

                    # kpt loss
                    if self.do_predict_kpts:
                        loss = self.calc_kpt_loss(batch_t, out)["loss"]
                        total_losses.append(loss)

                    continue

            if self.use_prev_pose_condition and self.do_predict_rel_pose:
                prev_pose = pose_prev_pred_abs

            pose_mat_gt_abs = torch.stack([self.pose_to_mat_converter_fn(rt) for rt in pose_gt_abs])
            if self.do_predict_rel_pose:
                t_gt_rel, rot_gt_rel_mat = pose_to_egocentric_delta_pose(pose_mat_prev_gt_abs, pose_mat_gt_abs)

            if stage != "train":
                pose_model = None
            elif self.do_predict_rel_pose:
                pose_model = torch.stack(
                    [torch.cat([r, t]) for r, t in zip(self.rot_mat_to_vector_converter_fn(rot_gt_rel_mat), t_gt_rel)]
                )
            else:
                pose_model = pose_gt_abs

            out = self.model(
                rgb,
                pose=pose_model,
                depth=depth,
                bbox=bbox_2d,
                prev_pose=prev_pose,
                prev_latent=prev_latent,
                state=state,
                features_rgb=features_rgb,
            )

            # POSTPROCESS OUTPUTS

            t_pred, rot_pred = average_pose(out["t"], self.rot_vector_to_mat_converter_fn(out["rot"]))
            state = out["state"]

            if self.do_predict_2d_t:
                center_depth_pred = out["center_depth"]
                convert_2d_t_pred_to_3d_res = convert_2d_t_to_3d(t_pred, center_depth_pred, intrinsics.float(), hw=hw)
                t_pred = convert_2d_t_pred_to_3d_res["t_pred"]

            rot_mat_gt_abs = pose_mat_gt_abs[:, :3, :3]

            # pose_mat_pred = torch.stack(
            #     [self.pose_to_mat_converter_fn(rt) for rt in torch.cat([t_pred, rot_pred], dim=1)]
            # )
            pose_mat_pred = torch.stack([convert_r_t_to_rt(r, t) for r, t in zip(rot_pred, t_pred)])

            if self.use_pnp_for_rot_pred:
                kpts = out["kpts"]
                rot_mat_pred_bidxs = []
                for bidx in range(batch_size):
                    prev_kpts_2d = prev_kpts[bidx]
                    prev_kpts_2d_denorm = prev_kpts_2d * torch.tensor([w, h]).to(self.device)
                    prev_depth = batched_seq[t - 1]["depth"][bidx]
                    prev_kpts_depth_actual = prev_depth[
                        ..., prev_kpts_2d_denorm[:, 1].long(), prev_kpts_2d_denorm[:, 0].long()
                    ].double()
                    prev_kpts_depth = batched_seq[t - 1]["bbox_2d_kpts_depth"][bidx] / 10
                    prev_visib_kpt_mask = torch.abs(prev_kpts_depth - prev_kpts_depth_actual) < 1e-2
                    prev_kpts_2d_denorm_visib = prev_kpts_2d_denorm[prev_visib_kpt_mask]
                    prev_kpts_visib_depth = prev_kpts_depth[prev_visib_kpt_mask]
                    prev_kpts_3d = backproj_2d_to_3d(prev_kpts_2d_denorm_visib, prev_kpts_visib_depth, intrinsics[bidx])
                    kpts_2d_denorm = kpts[bidx] * torch.tensor([w, h]).to(self.device)
                    visib_kpt_mask = prev_visib_kpt_mask
                    kpts_2d_denorm_visib = kpts_2d_denorm[visib_kpt_mask]
                    pose_from_3d_2d_matches_res = get_pose_from_3d_2d_matches(
                        prev_kpts_3d, kpts_2d_denorm_visib, intrinsics[bidx]
                    )
                    rot_mat_pred_bidx = pose_from_3d_2d_matches_res["R"]
                    rot_mat_pred_bidxs.append(torch.tensor(rot_mat_pred_bidx))

                pose_mat_pred[:, :3, :3] = torch.stack(rot_mat_pred_bidxs).to(self.device)

            rot_mat_pred = pose_mat_pred[:, :3, :3]

            if self.do_predict_rel_pose:
                pose_mat_prev_pred_abs = torch.stack(
                    [
                        self.pose_to_mat_converter_fn(rt)
                        for rt in torch.cat([pose_prev_pred_abs["t"], pose_prev_pred_abs["rot"]], dim=1)
                    ]
                )
                pose_mat_pred_abs = egocentric_delta_pose_to_pose(
                    pose_mat_prev_pred_abs,
                    trans_delta=pose_mat_pred[:, :3, 3],
                    rot_mat_delta=pose_mat_pred[:, :3, :3],
                    do_couple_rot_t=False,
                )
            else:
                pose_mat_pred_abs = pose_mat_pred

            t_pred_abs = pose_mat_pred_abs[:, :3, 3]
            rot_mat_pred_abs = pose_mat_pred_abs[:, :3, :3]

            # LOSSES
            # -- t_pred/rot_pred can be rel or abs

            if self.use_pose_loss:
                if self.do_predict_rel_pose:
                    pose_mat_gt_rel = convert_r_t_to_rt(rot_gt_rel_mat, t_gt_rel)
                    loss_pose = self.criterion_pose(pose_mat_pred, pose_mat_gt_rel, pts)
                else:
                    loss_pose = self.criterion_pose(pose_mat_pred_abs, pose_mat_gt_abs, pts)
                loss = loss_pose.clone()
            else:
                # t loss

                if self.do_predict_2d_t:
                    t_pred_2d = out["t"]
                    if self.do_predict_rel_pose:
                        loss_uv = self.criterion_trans(t_pred_2d[:, :2], t_gt_rel[:, :2])
                        # trickier for depth (should be change in scale)
                        loss_z = self.criterion_trans(center_depth_pred, t_gt_rel[:, 2:3])
                    else:
                        t_gt_2d_norm, depth_gt = convert_3d_t_for_2d(t_gt_abs, intrinsics, hw)
                        loss_uv = self.criterion_trans(t_pred_2d, t_gt_2d_norm)
                        loss_z = self.criterion_trans(center_depth_pred, depth_gt)
                    loss_t = loss_uv + loss_z
                else:
                    if self.do_predict_rel_pose:
                        loss_t = self.criterion_trans(out["t"], t_gt_rel.unsqueeze(1).expand_as(out["t"]))
                    else:
                        loss_t = self.criterion_trans(out["t"], t_gt_abs.unsqueeze(1).expand_as(out["t"]))

                # rot loss

                if self.criterion_rot_name == "displacement":
                    self.criterion_rot = functools.partial(self.criterion_rot, pts=pts)

                if self.do_predict_rel_pose:
                    if self.use_rot_mat_for_loss:
                        loss_rot = self.criterion_rot(rot_mat_pred, rot_gt_rel_mat)
                    else:
                        rot_gt_rel = self.rot_mat_to_vector_converter_fn(rot_gt_rel_mat)
                        loss_rot = self.criterion_rot(out["rot"], rot_gt_rel.unsqueeze(1).expand_as(out["rot"]))
                else:
                    if self.use_rot_mat_for_loss:
                        loss_rot = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        loss_rot = self.criterion_rot(out["rot"], rot_gt_abs.unsqueeze(1).expand_as(out["rot"]))

                # if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                #     loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)!
                #     loss_t += loss_t_abs / seq_length
                #     if self.use_rot_mat_for_loss:
                #         loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                #     else:
                #         rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                #         loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                #     loss_rot += loss_rot_abs / seq_length

                if self.opt_only is None:
                    loss = self.tf_t_loss_coef * loss_t + self.tf_rot_loss_coef * loss_rot
                else:
                    loss = 0
                    assert any(x in self.opt_only for x in ["rot", "t"]), f"Invalid opt_only: {self.opt_only}"
                    if "rot" in self.opt_only:
                        loss += loss_rot * self.tf_rot_loss_coef
                    if "t" in self.opt_only:
                        loss += loss_t * self.tf_t_loss_coef

            # depth loss
            if self.use_belief_decoder:
                loss_depth = F.mse_loss(out["decoder_out"]["depth_final"], out["latent_depth"])
            else:
                loss_depth = torch.tensor(0.0).to(self.device)
            loss += loss_depth

            # priv loss
            if "priv_decoded" in out:
                loss_priv = compute_chamfer_dist(out["priv_decoded"], batch_t["priv"])
                loss += loss_priv * 0.01

            # kpt loss
            if self.do_predict_kpts:
                calc_kpt_loss_res = self.calc_kpt_loss(batch_t, out)
                loss_kpts = calc_kpt_loss_res["loss_kpts"]
                loss_cr = calc_kpt_loss_res["loss_cr"]
                loss += calc_kpt_loss_res["loss"]

            if self.do_predict_rel_pose and self.do_predict_abs_pose:
                t_pred_abs_pose, rot_pred_abs_pose = out["t_abs_pose"], out["rot_abs_pose"]
                losses_abs_pose = self.calc_abs_pose_loss_for_rel(
                    pose_gt_abs, t_pred_abs_pose=t_pred_abs_pose, rot_pred_abs_pose=rot_pred_abs_pose
                )
                loss += losses_abs_pose["loss_abs_pose"]
                for k, v in losses_abs_pose.items():
                    seq_stats[k].append(v.item())

            kl_loss = kl_divergence(out["mu"], out["logvar"])
            # kept at 0 for 1k steps in the original paper
            loss += self.kl_loss_coef * kl_loss

            seq_stats["kl_loss"].append(kl_loss.item())
            kde_gaussian_sigma = 0.1
            if self.do_predict_rel_pose:
                t_log_likelihood = evaluate_tras_likelihood(t_gt_rel, out["t"], kde_gaussian_sigma)
            else:
                t_log_likelihood = evaluate_tras_likelihood(t_gt_abs, out["t"], kde_gaussian_sigma)
            seq_metrics["t_log_likelihood"].append(t_log_likelihood.item())

            # optim
            if do_opt_every_ts:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
                if do_vis:
                    grad_norms, grad_norm = self.get_grad_info()
                    vis_data["grad_norm"].append(grad_norm)
                    vis_data["grad_norms"].append(grad_norms)
                optimizer.step()
                optimizer.zero_grad()
                state = [x if x is None else x.detach() for x in state]
            elif do_opt_in_the_end:
                total_losses.append(loss)

            # METRICS

            if self.do_predict_rel_pose:
                pose_mat_pred_metrics = pose_mat_pred
                pose_mat_gt_metrics = convert_r_t_to_rt(rot_gt_rel_mat, t_gt_rel)
            else:
                pose_mat_pred_metrics = pose_mat_pred_abs
                pose_mat_gt_metrics = pose_mat_gt_abs

            m_batch_avg = self.calc_metrics_batch(
                batch_t, pose_mat_pred_metrics=pose_mat_pred_metrics, pose_mat_gt_metrics=pose_mat_gt_metrics
            )
            for k, v in m_batch_avg.items():
                seq_metrics[k].append(v)

            # OTHER

            seq_stats["loss"].append(loss.item())
            seq_stats["loss_depth"].append(loss_depth.item())
            if self.use_pose_loss:
                seq_stats["loss_pose"].append(loss_pose.item())
            else:
                seq_stats["loss_rot"].append(loss_rot.item())
                seq_stats["loss_t"].append(loss_t.item())
                if self.include_abs_pose_loss_for_rel and t == seq_length - 1:
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())
            if "priv_decoded" in out:
                seq_stats["loss_priv"].append(loss_priv.item())
            if self.do_predict_kpts:
                seq_stats["loss_kpts"].append(loss_kpts.item())
                seq_stats["loss_cr"].append(loss_cr.item())
            if self.do_predict_2d_t:
                seq_stats["loss_uv"].append(loss_uv.item())
                seq_stats["loss_z"].append(loss_z.item())

            if self.do_log and self.do_log_every_ts:
                for k, v in m_batch_avg.items():
                    self.writer.add_scalar(f"{stage}_ts/{k}", v, self.ts_counts_per_stage[stage])
            if self.do_debug:
                for k, v in {**m_batch_avg, **{"loss_rot": loss_rot.item(), "loss_t": loss_t.item()}}.items():
                    self.seq_stats_all_seq[self.seq_counts_per_stage[stage]][k].append(v)

            self.ts_counts_per_stage[stage] += 1

            if save_preds:
                assert preds_dir is not None, "preds_dir must be provided for saving predictions"
                save_results(batch_t, pose_mat_pred_abs, preds_dir, gt_pose=pose_mat_gt_abs)

            if do_vis:
                # save inputs to the exp dir
                vis_keys = ["rgb", "intrinsics"]
                for k in ["mask", "mesh_bbox", "pts", "depth"]:
                    if k in batch_t and len(batch_t[k]) > 0:
                        vis_keys.append(k)
                for k in vis_keys:
                    vis_data[k].append([detach_and_cpu(batch_t[k][i]) for i in vis_batch_idxs])
                vis_data["pose_mat_pred_abs"].append(detach_and_cpu(pose_mat_pred_abs[vis_batch_idxs]))
                vis_data["pose_mat_pred"].append(detach_and_cpu(pose_mat_pred[vis_batch_idxs]))
                vis_data["intrinsics"].append(detach_and_cpu(intrinsics[vis_batch_idxs]))
                vis_data["out"].append(
                    ({k: detach_and_cpu(v[vis_batch_idxs]) for k, v in out.items() if k in ["t", "rot"]})
                )
                vis_data["t_pred"].append(detach_and_cpu(t_pred[vis_batch_idxs]))
                vis_data["rot_pred"].append(detach_and_cpu(rot_pred[vis_batch_idxs]))
                vis_data["pose_mat_gt_abs"].append(detach_and_cpu(pose_mat_gt_abs[vis_batch_idxs]))

                if self.do_predict_2d_t:
                    if not self.do_predict_rel_pose:
                        vis_data["t_gt_2d_norm"].append(detach_and_cpu(t_gt_2d_norm))
                        vis_data["depth_gt"].append(detach_and_cpu(depth_gt))
                        vis_data["t_pred_2d_denorm"].append(
                            detach_and_cpu(convert_2d_t_pred_to_3d_res["t_pred_2d_denorm"])
                        )
                    vis_data["center_depth_pred"].append(detach_and_cpu(center_depth_pred))
                    vis_data["t_pred_2d"].append(detach_and_cpu(t_pred_2d))
                if "priv_decoded" in out:
                    vis_data["priv_decoded"].append(detach_and_cpu(out["priv_decoded"]))
                vis_data["pose_prev_pred_abs"].append(detach_and_cpu(pose_prev_pred_abs))
                vis_data["pts"].append(detach_and_cpu(pts))
                vis_data["bbox_3d"].append(detach_and_cpu(batch_t["mesh_bbox"]))
                vis_data["out_prev"].append(detach_and_cpu(out_prev))
                if self.do_predict_rel_pose:
                    if self.use_pose_loss:
                        vis_data["pose_mat_pred"].append(detach_and_cpu(pose_mat_pred))
                        vis_data["pose_mat_gt_rel"].append(detach_and_cpu(pose_mat_gt_rel))
                    else:
                        vis_data["t_gt_rel"].append(detach_and_cpu(t_gt_rel))
                        if not self.use_rot_mat_for_loss:
                            vis_data["rot_gt_rel"].append(detach_and_cpu(rot_gt_rel))
                        vis_data["rot_gt_rel_mat"].append(detach_and_cpu(rot_gt_rel_mat))
                        vis_data["pose_mat_prev_gt_abs"].append(detach_and_cpu(pose_mat_prev_gt_abs))
                if self.do_predict_kpts:
                    vis_data["kpts_pred"].append(detach_and_cpu(out["kpts"]))
                    vis_data["kpts_gt"].append(detach_and_cpu(batch_t["bbox_2d_kpts"]))
                    if self.use_pnp_for_rot_pred:
                        vis_data["prev_kpts"].append(detach_and_cpu(prev_kpts))

                if torch.isnan(loss):
                    if t > 0:
                        self.logger.error(f"{batched_seq[t-1]=}")
                    self.logger.error(f"{batched_seq[t]=}")
                    self.logger.error(f"{loss_t=}")
                    self.logger.error(f"{loss_rot=}")
                    self.logger.error(f"rot_pred: {rot_pred}")
                    self.logger.error(f"rot_mat_pred: {rot_mat_pred}")
                    self.logger.error(f"rot_gt_abs: {rot_gt_abs}")
                    self.logger.error(f"rot_mat_gt_abs: {rot_mat_gt_abs}")
                    self.logger.error(f"seq_metrics: {seq_metrics}")
                    self.logger.error(f"seq_stats: {seq_stats}")
                    if self.do_predict_rel_pose:
                        self.logger.error(f"rot_gt_rel: {rot_gt_rel_mat}")
                    sys.exit(1)

            # UPDATE VARS

            if self.do_predict_rel_pose:
                if self.do_predict_abs_pose:
                    pose_prev_pred_abs = {"t": t_pred_abs_pose, "rot": rot_pred_abs_pose}
                else:
                    rot_prev_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                    pose_prev_pred_abs = {"t": t_pred_abs, "rot": rot_prev_pred_abs}
            else:
                pose_prev_pred_abs = {"t": t_pred, "rot": rot_pred}

            pose_mat_prev_gt_abs = pose_mat_gt_abs
            out_prev = {"t": out["t"], "rot": out["rot"]}
            prev_latent = out["latent"] if self.use_prev_latent else None

        if do_opt_in_the_end:
            total_loss = torch.mean(torch.stack(total_losses))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_clip_grad_norm)
            if do_vis:
                grad_norms, grad_norm = self.get_grad_info()
                vis_data["grad_norm"].append(grad_norm)
                vis_data["grad_norms"].append(grad_norms)
            optimizer.step()
            optimizer.zero_grad()

        if self.use_rnn:
            last_step_state = {
                "prev_latent": prev_latent.detach() if self.use_prev_latent else None,
                "pose_prev_pred_abs": {k: v.detach() for k, v in pose_prev_pred_abs.items()},
                "pose_mat_prev_gt_abs": pose_mat_prev_gt_abs,
                "state": [x if x is None else x.detach() for x in state],
            }
        else:
            last_step_state = None

        for stats in [seq_stats, seq_metrics]:
            for k, v in stats.items():
                stats[k] = np.mean(v)

        if self.do_predict_rel_pose:
            # calc loss/metrics btw accumulated abs poses
            metrics_abs = self.calc_metrics_batch(batch_t, pose_mat_pred_abs, pose_mat_gt_abs)
            for k, v in metrics_abs.items():
                seq_metrics[f"{k}_abs"].append(v)
            if not self.include_abs_pose_loss_for_rel:
                with torch.no_grad():
                    loss_t_abs = self.criterion_trans(t_pred_abs, t_gt_abs)
                    if self.use_rot_mat_for_loss:
                        loss_rot_abs = self.criterion_rot(rot_mat_pred_abs, rot_mat_gt_abs)
                    else:
                        rot_pred_abs = self.rot_mat_to_vector_converter_fn(rot_mat_pred_abs)
                        loss_rot_abs = self.criterion_rot(rot_pred_abs, rot_gt_abs)
                    seq_stats["loss_t_abs"].append(loss_t_abs.item())
                    seq_stats["loss_rot_abs"].append(loss_rot_abs.item())

        if do_vis:
            os.makedirs(self.vis_dir, exist_ok=True)
            save_vis_path = (
                f"{self.vis_dir}/{stage}_epoch_{self.train_epoch_count}_step_{self.ts_counts_per_stage[stage]}.pt"
            )
            torch.save(vis_data, save_vis_path)
            self.save_vis_paths.append(save_vis_path)
            self.logger.info(f"Saved vis data for exp {Path(self.exp_dir).name} to {save_vis_path}")

        return {
            "losses": seq_stats,
            "metrics": seq_metrics,
            "last_step_state": last_step_state,
        }


def average_pose(tra_hat: torch.Tensor, rot_hat: torch.Tensor) -> torch.Tensor:
    """Compute the average pose of a number of samples.

    L2 mean is computed for translations and chordal L2 mean for rotations.
    Ref "Rotation averaging" by Hartley et al (2013).

    Args:
        tra_hat: translation samples, shape (N, M, 3)
        rot_hat: rotation matrix samples, shape (N, M, 3, 3)

    Returns:
        average translation, shape (N, 3),
        average rotation, shape (N, 3, 3)
    """
    tra = torch.mean(tra_hat, dim=1)
    rot = torch.cat([chordal_l2_mean(R)[None, :, :] for R in rot_hat])

    return tra, rot


def chordal_l2_mean(rot_samples: torch.Tensor) -> torch.Tensor:
    """Compute the a single rotation average of many samples.

    Chordal L2 mean for rotations as done in "Rotation averaging" Hartley et al (2013).

    Args:
        rot_samples: rotation matrix samples, shape (N, 3, 3)

    Returns:
        average rotation, shape (3, 3)
    """
    C = torch.sum(rot_samples, dim=0)
    U, _, Vt = torch.linalg.svd(C)

    S = U @ Vt
    if torch.linalg.det(S) < 0:
        S = U @ torch.diag(torch.tensor([1.0, 1.0, -1.0], device=rot_samples.device, dtype=rot_samples.dtype)) @ Vt

    return S


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute the KL divergence to a standard Gaussian.

    Computes the mean KL-divergence for a set of Gaussian distributions with diagonal
    covariance matrices (M-variate).

    Args:
        mu: mean of the distribution, shape (N, M)
        logvar: log of variance of the distribution, shape (N, M)

    Returns:
        mean KL divergence
    """
    return torch.mean(torch.sum(0.5 * (torch.exp(logvar) + mu**2 - 1 - logvar), dim=1))


def evaluate_tras_likelihood(queries: torch.Tensor, samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the log probability densities of query points based on dist samples.

    Args:
        queries: query translations, shape (N, 3)
        samples: samples drawn from distributions, shape (N, M, 3)
        sigma: bandwidth parameter used for density estimation

    Returns:
        mean of log likelihoods of query points
    """
    return torch.mean(
        torch.tensor([get_tra_log_likelihood(query, sample, sigma).item() for query, sample in zip(queries, samples)])
    )


def evaluate_rots_likelihood(queries: torch.Tensor, samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the log probability densities of query points based on dist samples.

    Args:
        queries: query rotations in quaternion param [w, x, y, z], shape (N, 4)
        samples: samples in quat param [w, x, y, z] drawn from dist, shape (N, M, 4)
        sigma: bandwidth parameter used for density estimation

    Returns:
        mean of log likelihoods of query points
    """
    return torch.mean(
        torch.tensor([get_rot_log_likelihood(query, sample, sigma).item() for query, sample in zip(queries, samples)])
    )


def get_tra_log_likelihood(query: torch.Tensor, samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the log probability density of a query point based on dist samples.

    Args:
        query: query point, shape (3,)
        samples: samples drawn from distribution, shape (N, 3)
        sigma: bandwidth parameter used for density estimation

    Returns:
        log likelihood of query point
    """
    dist = R3Gaussian(samples, sigma)
    return dist.log_prob(query.float())


def get_rot_log_likelihood(query: torch.Tensor, samples: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute the log probability density of a query point based on dist samples.

    Args:
        query: query point in quaternion parameterization [w, x, y, z], shape (4,)
        samples:
            samples in quaternion parameterization [w, x, y, z] drawn from distribution
            shape (N, 4)
        sigma: bandwidth parameter used for density estimation

    Returns:
        log likelihood of query point
    """
    dist = SO3Bingham(samples, sigma)
    return dist.log_prob(query[None, :])


class R3Gaussian:
    """Approximation of a trivariate distribution with a Gaussian kernel.

    KDE is applied on n i.i.d. samples from an unknown distribution to be modelled,
    using a zero-mean isotropic Gaussian kernel with tunable smooting parameter.
    """

    def __init__(self, samples: torch.Tensor, sigma: float) -> None:
        """Construct the approximation of the distribution.

        Args:
            samples: samples drawn from the dsitribution to be modelled, shape (N, 3)
            sigma: standard deviation of Gaussian kernel along all directions
        """
        n = len(samples)
        mix = D.Categorical(torch.ones(n, device=samples.device))
        comp = D.MultivariateNormal(
            samples,
            (sigma**2 * torch.eye(3, device=samples.device)).unsqueeze(0).repeat(n, 1, 1),
        )
        self._gmm = D.MixtureSameFamily(mix, comp)

    def sample(self, sample_shape: Tuple = torch.Size) -> torch.Tensor:
        """Draw samples from the modelled distribution.

        Args:
            sample_shape: shape of samples to be drawn

        Returns:
            drawn samples
        """
        return self._gmm.sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate log probability of the distribution.

        Args:
            value: value(s) at which the density is evaluated

        Returns:
            log probability at the queried values
        """
        return self._gmm.log_prob(value)
