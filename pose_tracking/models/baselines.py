import torch
from pose_tracking.models.cnnlstm import MLP
from pose_tracking.models.cvae import get_encoders
from pose_tracking.utils.misc import init_params
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.encoder_name = encoder_name
        self.encoder_img_weights = encoder_img_weights
        self.norm_layer_type = norm_layer_type
        self.rt_hidden_dim = rt_hidden_dim

        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_depth = use_depth

        self.input_dim = encoder_out_dim * 2 if use_depth else encoder_out_dim

        self.encoder_img, self.encoder_depth = get_encoders(
            encoder_name,
            do_freeze=False,
            weights_rgb=encoder_img_weights,
            norm_layer_type=norm_layer_type,
            out_dim=encoder_out_dim,
        )

        if not self.use_depth:
            self.encoder_depth = None

        self.t_mlp_in_dim = self.rot_mlp_in_dim = encoder_out_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = encoder_out_dim
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

        res["latent"] = latent
        res["state"] = [None]

        latent_t = latent
        latent_rot = latent

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
            latent_t = torch.cat([latent_t, t_prev], dim=1)
            latent_rot = torch.cat([latent_rot, rot_prev], dim=1)
        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            latent_t = torch.cat([latent_t, prev_latent], dim=1)
            latent_rot = torch.cat([latent_rot, prev_latent], dim=1)

        t_in = latent_t
        rot_in = latent_rot

        if self.do_predict_t:
            t = self.t_mlp(t_in)
        else:
            t = torch.zeros(bs, self.t_mlp_out_dim, device=rgb.device)
        res["t"] = t

        if self.do_predict_rot:
            rot = self.rot_mlp(rot_in)
        else:
            rot = torch.zeros(bs, self.rot_mlp_out_dim, device=rgb.device)
        res["rot"] = rot

        return res


class KeypointCNN(nn.Module):
    def __init__(
        self,
        bbox_num_kpts=32,
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
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.encoder_name = encoder_name
        self.encoder_img_weights = encoder_img_weights
        self.norm_layer_type = norm_layer_type
        self.rt_hidden_dim = rt_hidden_dim
        self.num_kpts = bbox_num_kpts

        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_depth = use_depth

        self.input_dim = encoder_out_dim * 2 if use_depth else encoder_out_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)
        self.t_mlp_out_dim = 2 if do_predict_2d_t else 3

        self.encoder_img, self.encoder_depth = get_encoders(
            encoder_name,
            do_freeze=False,
            weights_rgb=encoder_img_weights,
            norm_layer_type=norm_layer_type,
            out_dim=encoder_out_dim,
            rgb_in_channels=3,
        )

        if not self.use_depth:
            self.encoder_depth = None

        self.t_mlp_in_dim = self.rot_mlp_in_dim = encoder_out_dim

        if use_prev_latent:
            self.t_mlp_in_dim += self.input_dim

        self.t_mlp = MLP(
            in_dim=self.t_mlp_in_dim,
            out_dim=bbox_num_kpts * 3,
            hidden_dim=self.rt_hidden_dim,
            num_layers=rt_mlps_num_layers,
            act_out=None,
            dropout=dropout_heads,
        )

    def forward(
        self,
        rgb,
        rgb_prev,
        depth=None,
        prev_pose=None,
        latent_rgb=None,
        latent_depth=None,
        prev_latent=None,
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

        res["latent"] = latent
        res["state"] = [None]

        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            latent = torch.cat([latent, prev_latent], dim=1)

        kpt_in = latent

        kpts = self.t_mlp(kpt_in).reshape(bs, self.num_kpts, 3)
        res["kpts"] = kpts

        t = torch.zeros(bs, self.t_mlp_out_dim, device=rgb.device)
        rot = torch.zeros(bs, self.rot_mlp_out_dim, device=rgb.device)
        res["t"] = t
        res["rot"] = rot

        return res


class KeypointPose(nn.Module):
    def __init__(
        self,
        bbox_num_kpts=32,
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
        use_both_kpts_as_input=False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.encoder_name = encoder_name
        self.encoder_img_weights = encoder_img_weights
        self.norm_layer_type = norm_layer_type
        self.rt_hidden_dim = rt_hidden_dim
        self.num_kpts = bbox_num_kpts

        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_depth = use_depth
        self.use_both_kpts_as_input = use_both_kpts_as_input

        self.input_dim = encoder_out_dim

        self.t_mlp_in_dim = self.rot_mlp_in_dim = encoder_out_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = encoder_out_dim
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

        self.shared_mlp = MLP(
            in_dim=self.num_kpts * 3 * (2 if use_both_kpts_as_input else 1),
            out_dim=self.encoder_out_dim,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            dropout=dropout_heads,
        )

        init_params(self)

    def forward(
        self,
        bbox_kpts,
        prev_bbox_kpts=None,
        prev_pose=None,
        prev_latent=None,
        **kwargs,
    ):

        bs, n, _ = bbox_kpts.size()
        if prev_bbox_kpts is None and not self.use_both_kpts_as_input:
            shared_mlp_in = bbox_kpts
        else:
            shared_mlp_in = torch.cat([bbox_kpts, prev_bbox_kpts], dim=1)
        latent = self.shared_mlp(shared_mlp_in.view(bs, -1))

        res = {}
        res["latent"] = latent
        res["state"] = [None]

        latent_t = latent
        latent_rot = latent

        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_pose = {
                    "t": torch.zeros(
                        bs,
                        self.t_mlp_out_dim + 1 if self.do_predict_2d_t else self.t_mlp_out_dim,
                        device=bbox_kpts.device,
                    ),
                    "rot": torch.zeros(bs, self.rot_mlp_out_dim, device=bbox_kpts.device),
                }
            if self.do_predict_2d_t:
                t_prev = prev_pose["t"][:, :2]
            else:
                t_prev = prev_pose["t"]
            rot_prev = prev_pose["rot"]
            if self.use_mlp_for_prev_pose:
                t_prev = self.prev_t_mlp(t_prev)
                rot_prev = self.prev_rot_mlp(prev_pose["rot"])
            latent_t = torch.cat([latent_t, t_prev], dim=1)
            latent_rot = torch.cat([latent_rot, rot_prev], dim=1)
        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            latent_t = torch.cat([latent_t, prev_latent], dim=1)
            latent_rot = torch.cat([latent_rot, prev_latent], dim=1)

        t_in = latent_t
        rot_in = latent_rot

        if self.do_predict_t:
            t = self.t_mlp(t_in)
        else:
            t = torch.zeros(bs, self.t_mlp_out_dim, device=bbox_kpts.device)
        res["t"] = t

        if self.do_predict_rot:
            rot = self.rot_mlp(rot_in)
        else:
            rot = torch.zeros(bs, self.rot_mlp_out_dim, device=bbox_kpts.device)
        res["rot"] = rot

        return res


class KeypointKeypoint(nn.Module):
    def __init__(
        self,
        bbox_num_kpts=32,
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
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.encoder_out_dim = encoder_out_dim
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.encoder_name = encoder_name
        self.encoder_img_weights = encoder_img_weights
        self.norm_layer_type = norm_layer_type
        self.rt_hidden_dim = rt_hidden_dim
        self.num_kpts = bbox_num_kpts

        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_depth = use_depth

        self.input_dim = encoder_out_dim

        self.input_dim = encoder_out_dim * 2 if use_depth else encoder_out_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)
        self.t_mlp_out_dim = 2 if do_predict_2d_t else 3

        self.t_mlp_in_dim = self.rot_mlp_in_dim = encoder_out_dim

        if use_prev_latent:
            self.t_mlp_in_dim += self.input_dim

        self.t_mlp = MLP(
            in_dim=self.t_mlp_in_dim,
            out_dim=bbox_num_kpts * 3,
            hidden_dim=self.rt_hidden_dim,
            num_layers=rt_mlps_num_layers,
            act_out=None,
            dropout=dropout_heads,
        )

        self.shared_mlp = MLP(
            in_dim=self.num_kpts * 3 * 2,
            out_dim=self.encoder_out_dim,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            dropout=dropout_heads,
            act_out=nn.LeakyReLU(),
        )

        init_params(self)

    def forward(
        self,
        bbox_kpts,
        prev_bbox_kpts=None,
        prev_pose=None,
        prev_latent=None,
        **kwargs,
    ):

        bs, n, _ = bbox_kpts.size()
        if prev_bbox_kpts is None:
            prev_bbox_kpts = torch.zeros_like(bbox_kpts)

        shared_in = torch.cat([bbox_kpts, prev_bbox_kpts], dim=1).view(bs, -1)
        latent = self.shared_mlp(shared_in)

        res = {}
        res["latent"] = latent
        res["state"] = [None]

        latent_t = latent.clone()
        latent_rot = latent.clone()

        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_pose = {
                    "t": torch.zeros(
                        bs,
                        self.t_mlp_out_dim + 1 if self.do_predict_2d_t else self.t_mlp_out_dim,
                        device=bbox_kpts.device,
                    ),
                    "rot": torch.zeros(bs, self.rot_mlp_out_dim, device=bbox_kpts.device),
                }
            if self.do_predict_2d_t:
                t_prev = prev_pose["t"][:, :2]
            else:
                t_prev = prev_pose["t"]
            rot_prev = prev_pose["rot"]
            if self.use_mlp_for_prev_pose:
                t_prev = self.prev_t_mlp(t_prev)
                rot_prev = self.prev_rot_mlp(prev_pose["rot"])
            latent_t = torch.cat([latent_t, t_prev], dim=1)
            latent_rot = torch.cat([latent_rot, rot_prev], dim=1)
        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            latent_t = torch.cat([latent_t, prev_latent], dim=1)
            latent_rot = torch.cat([latent_rot, prev_latent], dim=1)

        res["latent"] = latent
        res["state"] = [None]

        if self.use_prev_latent:
            if prev_latent is None:
                prev_latent = torch.zeros_like(latent)
            latent = torch.cat([latent, prev_latent], dim=1)

        kpt_in = latent

        kpts = self.t_mlp(kpt_in).reshape(bs, self.num_kpts, 3)
        res["kpts"] = kpts

        t = torch.zeros(bs, self.t_mlp_out_dim, device=bbox_kpts.device)
        rot = torch.zeros(bs, self.rot_mlp_out_dim, device=bbox_kpts.device)
        res["t"] = t
        res["rot"] = rot

        return res
