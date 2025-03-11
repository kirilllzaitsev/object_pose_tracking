from typing import Tuple

import torch
from pose_tracking.models.cnnlstm import MLP, get_encoders
from torch import nn
from torchvision import models


class CVAE(nn.Module):
    def __init__(
        self,
        z_dim,
        hidden_dim=512,
        rt_mlps_num_layers=1,
        encoder_out_dim=512,
        dropout=0.0,
        dropout_heads=0.0,
        encoder_name="resnet18",
        encoder_img_weights="imagenet",
        norm_layer_type="frozen_bn",
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        do_predict_3d_rot=False,
        use_prev_pose_condition=False,
        use_prev_latent=False,
        do_predict_rot=True,
        do_predict_t=True,
        rt_hidden_dim=512,
        use_mlp_for_prev_pose=False,
        use_depth=False,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.encoder_name = encoder_name
        self.encoder_img_weights = encoder_img_weights
        self.norm_layer_type = norm_layer_type
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.rt_hidden_dim = rt_hidden_dim
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_depth = use_depth

        self.encoder_img, _ = get_encoders(
            encoder_name,
            do_freeze=False,
            weights_rgb=encoder_img_weights,
            norm_layer_type=norm_layer_type,
            out_dim=encoder_out_dim,
        )
        self.mu = MLP(in_dim=encoder_out_dim, out_dim=z_dim, hidden_dim=hidden_dim)
        self.logvar = MLP(in_dim=encoder_out_dim, out_dim=z_dim, hidden_dim=hidden_dim)

        self.t_mlp_in_dim = self.rot_mlp_in_dim = z_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = z_dim
            self.depth_mlp_out_dim = 1
            if use_prev_pose_condition:
                self.depth_mlp_in_dim += self.depth_mlp_out_dim
            if use_prev_latent:
                self.depth_mlp_in_dim += self.encoder_out_dim * 2 if use_depth else self.encoder_out_dim
            self.depth_mlp = MLP(
                in_dim=self.depth_mlp_in_dim,
                out_dim=self.depth_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout_heads,
            )
        else:
            self.t_mlp_out_dim = 3

        if use_prev_pose_condition:
            if use_mlp_for_prev_pose:
                self.prev_t_mlp = MLP(
                    in_dim=self.t_mlp_out_dim,
                    out_dim=self.rt_hidden_dim,
                    hidden_dim=self.rt_hidden_dim,
                    num_layers=rt_mlps_num_layers,
                    dropout=dropout,
                )
                self.prev_rot_mlp = MLP(
                    in_dim=self.rot_mlp_out_dim,
                    out_dim=self.rt_hidden_dim,
                    hidden_dim=self.rt_hidden_dim,
                    num_layers=rt_mlps_num_layers,
                    dropout=dropout,
                )
                self.t_mlp_in_dim += self.rt_hidden_dim
                self.rot_mlp_in_dim += self.rt_hidden_dim
            else:
                self.t_mlp_in_dim += self.t_mlp_out_dim
                self.rot_mlp_in_dim += self.rot_mlp_out_dim
        if use_prev_latent:
            self.t_mlp_in_dim += self.input_dim
            self.rot_mlp_in_dim += self.input_dim

        if do_predict_t:
            self.t_mlp = MLP(
                in_dim=self.t_mlp_in_dim,
                out_dim=self.t_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                act_out=nn.Sigmoid() if do_predict_2d_t else None,  # normalized coords
                dropout=dropout_heads,
            )
        if do_predict_rot:
            self.rot_mlp = MLP(
                in_dim=self.rot_mlp_in_dim,
                out_dim=self.rot_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout_heads,
            )

    def forward(
        self,
        rgb,
        depth=None,
        prev_pose=None,
        latent_rgb=None,
        latent_depth=None,
        prev_latent=None,
        state=None,
        features_rgb=None,
        **kwargs,
    ):

        if features_rgb is None:
            latent_rgb = self.encoder_img(rgb) if latent_rgb is None else latent_rgb
        else:
            latent_rgb = features_rgb
        if self.use_depth:
            latent_depth = self.encoder_depth(depth) if latent_depth is None else latent_depth

        bs = latent_rgb.size(0)
        res = {}
        if self.use_depth:
            latent = torch.cat([latent_rgb, latent_depth], dim=1)
        else:
            latent = latent_rgb
        extracted_obs = latent

        t_in = extracted_obs
        rot_in = extracted_obs
        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_pose = {
                    "t": torch.zeros(
                        bs,
                        self.t_mlp_out_dim + 1 if self.do_predict_2d_t else self.t_mlp_out_dim,
                        device=latent_rgb.device,
                    ),
                    "rot": torch.zeros(bs, self.rot_mlp_out_dim, device=latent_rgb.device),
                }
            if self.do_predict_2d_t:
                t_prev = prev_pose["t"][:, :2]
            else:
                t_prev = prev_pose["t"]
            rot_prev = prev_pose["rot"]
            if self.use_mlp_for_prev_pose:
                t_prev = self.prev_t_mlp(t_prev)
                rot_prev = self.prev_rot_mlp(prev_pose["rot"])
            t_in = torch.cat([t_in, t_prev], dim=1)
            rot_in = torch.cat([rot_in, rot_prev], dim=1)
        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            t_in = torch.cat([t_in, prev_latent], dim=1)
            rot_in = torch.cat([rot_in, prev_latent], dim=1)

        lat_mu = self.mu(latent)
        lat_logvar = self.logvar(latent)

        lat_std = torch.exp(0.5 * lat_logvar)
        eps = torch.randn((bs, self.num_samples, self.z_dim), device=rgb.device)
        lat_sample = eps * lat_std.unsqueeze(1) + lat_mu.unsqueeze(1)
        t_in = lat_sample.flatten(end_dim=1)
        rot_in = lat_sample.flatten(end_dim=1)

        t = self.t_mlp(t_in)
        res["t"] = t

        rot = self.rot_mlp(rot_in)
        res["rot"] = rot

        res["mu"] = lat_mu
        res["logvar"] = lat_logvar

        return res
