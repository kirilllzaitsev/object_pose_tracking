import copy

import torch
import torch.nn as nn
from pose_tracking.models.encoders import get_encoders
from pose_tracking.utils.misc import print_cls
from torchvision.ops import roi_align


class BeliefEncoder(nn.Module):
    def __init__(
        self,
        state_cell,
        state_cell_out_dim,
        depth_latent_dim,
        belief_enc_hidden_dim,
        belief_depth_enc_hidden_dim,
        belief_enc_num_layers=2,
        belief_depth_enc_num_layers=2,
        use_rnn=True,
        dropout=0.0,
    ):
        super().__init__()
        self.state_cell = state_cell
        self.depth_latent_dim = depth_latent_dim
        self.use_rnn = use_rnn

        self.belief_prior_mlp = MLP(
            in_dim=state_cell_out_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_enc_hidden_dim,
            num_layers=belief_enc_num_layers,
            dropout=dropout,
        )
        self.belief_depth_mlp = MLP(
            in_dim=state_cell_out_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_depth_enc_hidden_dim,
            num_layers=belief_depth_enc_num_layers,
            dropout=dropout,
        )

    def forward(self, latent_rgb, latent_depth, hx, cx=None):
        latent_obs = torch.cat([latent_rgb, latent_depth], dim=-1)
        if self.use_rnn:
            if cx is None:
                cell_out = self.state_cell(latent_obs, hx)
            else:
                cell_out = self.state_cell(latent_obs, (hx, cx))
        else:
            cell_out = self.state_cell(latent_obs)

        if isinstance(cell_out, dict):
            hx_new = cell_out["hidden_state"]
            cx_new = cell_out.get("cell_state")
        elif len(cell_out) == 2 and len(cell_out[0].shape) == 2:
            hx_new = cell_out[0]
            cx_new = cell_out[1]
        else:
            hx_new = cell_out
            cx_new = None

        prior_belief = hx_new
        prior_belief_encoded = self.belief_prior_mlp(prior_belief)
        prior_belief_depth_encoded = self.belief_depth_mlp(prior_belief)
        latent_depth_gated = latent_depth * torch.sigmoid(prior_belief_depth_encoded)
        posterior_belief = prior_belief_encoded + latent_depth_gated
        return {
            "posterior_belief": posterior_belief,
            "prior_belief": prior_belief,
            "belief_state": posterior_belief,
            "prior_belief_encoded": prior_belief_encoded,
            "prior_belief_depth_encoded": prior_belief_depth_encoded,
            "latent_depth_gated": latent_depth_gated,
            "hx": hx_new,
            "cx": cx_new,
        }

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())


class BeliefDecoder(nn.Module):
    def __init__(
        self,
        state_dim,
        priv_decoder_out_dim,
        depth_decoder_hidden_dim,
        priv_decoder_hidden_dim,
        depth_decoder_out_dim,
        hidden_attn_hidden_dim,
        priv_decoder_num_layers=1,
        depth_decoder_num_layers=1,
        hidden_attn_num_layers=1,
        use_priv_decoder=False,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_attn = MLP(
            in_dim=state_dim,
            out_dim=depth_decoder_out_dim,
            hidden_dim=hidden_attn_hidden_dim,
            num_layers=hidden_attn_num_layers,
            dropout=dropout,
            act_out=nn.Sigmoid(),
        )
        self.depth_decoder = MLP(
            in_dim=state_dim,
            out_dim=depth_decoder_out_dim,
            hidden_dim=depth_decoder_hidden_dim,
            num_layers=depth_decoder_num_layers,
            dropout=dropout,
        )
        self.use_priv_decoder = use_priv_decoder
        if self.use_priv_decoder:
            self.priv_decoder = MLP(
                in_dim=state_dim,
                out_dim=priv_decoder_out_dim,
                hidden_dim=priv_decoder_hidden_dim,
                num_layers=priv_decoder_num_layers,
                dropout=dropout,
            )

    def forward(self, ht, depth_latent):
        attn = self.hidden_attn(ht)
        depth_latent_attn = depth_latent * attn
        depth_decoded = self.depth_decoder(ht)
        depth_final = depth_latent_attn + depth_decoded
        res = {
            "attn": attn,
            "depth_latent_attn": depth_latent_attn,
            "depth_decoded": depth_decoded,
            "depth_final": depth_final,
        }
        if self.use_priv_decoder:
            priv_decoded = self.priv_decoder(ht)
            priv_decoded = priv_decoded.view(-1, 256, 3)
            res["priv_decoded"] = priv_decoded

        return res

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1, act="leaky_relu", act_out=None, dropout=0.0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        if num_layers > 1:
            self.layers = []
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for i in range(num_layers - 2):
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.layers = [nn.Linear(in_dim, out_dim)]
        self.layers = nn.ModuleList(self.layers)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "leaky_relu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function {act}")
        self.act_out = act_out

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        if self.act_out is not None:
            x = self.act_out(x)
        return x

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())


class RecurrentCNN(nn.Module):
    def __init__(
        self,
        depth_dim,
        rgb_dim,
        hidden_dim,
        benc_belief_enc_hidden_dim,
        benc_belief_depth_enc_hidden_dim,
        bdec_priv_decoder_out_dim,
        bdec_priv_decoder_hidden_dim,
        bdec_depth_decoder_hidden_dim,
        bdec_hidden_attn_hidden_dim,
        benc_belief_enc_num_layers=2,
        benc_belief_depth_enc_num_layers=2,
        priv_decoder_num_layers=1,
        depth_decoder_num_layers=1,
        hidden_attn_num_layers=1,
        rt_mlps_num_layers=1,
        encoder_out_dim=256,
        dropout=0.0,
        dropout_heads=0.0,
        rnn_type="gru",
        encoder_name="resnet18",
        encoder_img_weights="imagenet",
        encoder_depth_weights=None,
        norm_layer_type="frozen_bn",
        rnn_state_init_type="zeros",
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        do_predict_3d_rot=False,
        use_rnn=True,
        use_obs_belief=False,
        use_priv_decoder=False,
        use_mlp_for_prev_pose=False,
        do_freeze_encoders=False,
        use_prev_pose_condition=False,
        use_prev_latent=False,
        use_belief_decoder=True,
        do_predict_kpts=False,
        do_predict_rot=True,
        r_num_layers_inc=0,
        rt_hidden_dim=None,
    ):
        super().__init__()
        self.depth_dim = depth_dim
        self.rgb_dim = rgb_dim
        self.hidden_dim = hidden_dim
        self.benc_belief_enc_hidden_dim = benc_belief_enc_hidden_dim
        self.benc_belief_depth_enc_hidden_dim = benc_belief_depth_enc_hidden_dim
        self.bdec_priv_decoder_out_dim = bdec_priv_decoder_out_dim
        self.bdec_priv_decoder_hidden_dim = bdec_priv_decoder_hidden_dim
        self.bdec_depth_decoder_hidden_dim = bdec_depth_decoder_hidden_dim
        self.bdec_hidden_attn_hidden_dim = bdec_hidden_attn_hidden_dim
        self.encoder_out_dim = encoder_out_dim
        self.benc_belief_enc_num_layers = benc_belief_enc_num_layers
        self.benc_belief_depth_enc_num_layers = benc_belief_depth_enc_num_layers
        self.rnn_type = rnn_type
        self.encoder_name = encoder_name
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.priv_decoder_num_layers = priv_decoder_num_layers
        self.depth_decoder_num_layers = depth_decoder_num_layers
        self.hidden_attn_num_layers = hidden_attn_num_layers
        self.rnn_state_init_type = rnn_state_init_type

        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.use_obs_belief = use_obs_belief
        self.use_belief_decoder = use_belief_decoder
        self.use_priv_decoder = use_priv_decoder
        self.use_rnn = use_rnn
        self.do_freeze_encoders = do_freeze_encoders
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_kpts = do_predict_kpts
        self.do_predict_rot = do_predict_rot
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose

        self.input_dim = depth_dim + rgb_dim
        self.rt_hidden_dim = hidden_dim if rt_hidden_dim is None else rt_hidden_dim

        if use_obs_belief:
            if use_rnn:
                if rnn_type == "lstm":
                    self.state_cell = nn.LSTMCell(self.input_dim, hidden_dim)
                else:
                    self.state_cell = nn.GRUCell(self.input_dim, hidden_dim)

                for name, param in self.state_cell.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)
            else:
                self.state_cell = MLP(
                    in_dim=self.input_dim,
                    out_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=dropout,
                )

            self.belief_encoder = BeliefEncoder(
                state_cell=self.state_cell,
                state_cell_out_dim=hidden_dim,
                depth_latent_dim=depth_dim,
                belief_enc_hidden_dim=benc_belief_enc_hidden_dim,
                belief_depth_enc_hidden_dim=benc_belief_depth_enc_hidden_dim,
                belief_enc_num_layers=benc_belief_enc_num_layers,
                belief_depth_enc_num_layers=benc_belief_depth_enc_num_layers,
                use_rnn=use_rnn,
                dropout=dropout,
            )
            if use_belief_decoder:
                self.belief_decoder = BeliefDecoder(
                    state_dim=hidden_dim,
                    priv_decoder_out_dim=bdec_priv_decoder_out_dim,
                    priv_decoder_hidden_dim=bdec_priv_decoder_hidden_dim,
                    depth_decoder_out_dim=depth_dim,
                    depth_decoder_hidden_dim=bdec_depth_decoder_hidden_dim,
                    hidden_attn_hidden_dim=bdec_hidden_attn_hidden_dim,
                    priv_decoder_num_layers=priv_decoder_num_layers,
                    depth_decoder_num_layers=depth_decoder_num_layers,
                    hidden_attn_num_layers=hidden_attn_num_layers,
                    use_priv_decoder=use_priv_decoder,
                    dropout=dropout,
                )
            else:
                self.belief_decoder = None
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = depth_dim + rgb_dim
            self.depth_mlp_out_dim = 1
            if use_prev_pose_condition:
                self.depth_mlp_in_dim += self.depth_mlp_out_dim
            if use_prev_latent:
                self.depth_mlp_in_dim += self.encoder_out_dim * 2
            self.depth_mlp = MLP(
                in_dim=self.depth_mlp_in_dim,
                out_dim=self.depth_mlp_out_dim,
                hidden_dim=hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout,
            )
        else:
            self.t_mlp_out_dim = 3

        self.t_mlp_in_dim = self.rot_mlp_in_dim = depth_dim + rgb_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)

        if use_prev_pose_condition:
            if use_mlp_for_prev_pose:
                self.prev_t_mlp = MLP(
                    in_dim=self.t_mlp_out_dim,
                    out_dim=hidden_dim,
                    hidden_dim=self.rt_hidden_dim,
                    num_layers=rt_mlps_num_layers,
                    dropout=dropout,
                )
                self.prev_rot_mlp = MLP(
                    in_dim=self.rot_mlp_out_dim,
                    out_dim=hidden_dim,
                    hidden_dim=self.rt_hidden_dim,
                    num_layers=rt_mlps_num_layers,
                    dropout=dropout,
                )
                self.t_mlp_in_dim += hidden_dim
                self.rot_mlp_in_dim += hidden_dim
            else:
                self.t_mlp_in_dim += self.t_mlp_out_dim
                self.rot_mlp_in_dim += self.rot_mlp_out_dim
        if use_prev_latent:
            self.t_mlp_in_dim += depth_dim + rgb_dim
            self.rot_mlp_in_dim += depth_dim + rgb_dim

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
                num_layers=rt_mlps_num_layers + r_num_layers_inc,
                dropout=dropout_heads,
            )

        if self.do_predict_kpts:
            self.kpts_mlp_in_dim = depth_dim + rgb_dim
            self.kpts_mlp_out_dim = (8 + 24) * 2
            self.kpts_mlp = MLP(
                in_dim=self.kpts_mlp_in_dim,
                out_dim=self.kpts_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout_heads,
            )

        self.encoder_name = encoder_name
        if encoder_name is None:
            self.encoder_img = self.encoder_depth = None
        else:
            self.encoder_img, self.encoder_depth = get_encoders(
                encoder_name,
                do_freeze=do_freeze_encoders,
                weights_rgb=encoder_img_weights,
                weights_depth=encoder_depth_weights,
                norm_layer_type=norm_layer_type,
                out_dim=encoder_out_dim,
            )

        self.hx = None
        self.cx = None

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())

    def reset_state(self, batch_size, device):
        if self.rnn_state_init_type == "learned":
            self.hx = nn.Parameter(torch.randn(1, self.hidden_dim, device=device))
            self.cx = None if "gru" in self.rnn_type else nn.Parameter(torch.randn(1, self.hidden_dim, device=device))
        elif self.rnn_state_init_type == "zeros":
            self.hx = torch.zeros(1, self.hidden_dim, device=device)
            self.cx = None if "gru" in self.rnn_type else torch.zeros(1, self.hidden_dim, device=device)
        elif self.rnn_state_init_type == "rand":
            self.hx = torch.randn(1, self.hidden_dim, device=device)
            self.cx = None if "gru" in self.rnn_type else torch.randn(1, self.hidden_dim, device=device)
        else:
            raise ValueError(f"Unknown rnn_state_init_type: {self.rnn_state_init_type}")

    def set_state(self, state):
        assert state is not None
        self.hx, self.cx = state

    def detach_state(self):
        if self.training:
            self.hx = self.hx.detach()
            if self.cx is not None:
                self.cx = self.cx.detach()

    def forward(
        self, rgb, depth, prev_pose=None, latent_rgb=None, latent_depth=None, prev_latent=None, state=None, **kwargs
    ):

        latent_rgb = self.encoder_img(rgb) if latent_rgb is None else latent_rgb
        latent_depth = self.encoder_depth(depth) if latent_depth is None else latent_depth

        res = {}
        state_new = None
        if self.use_obs_belief:
            if state is None:
                hx, cx = (self.hx, self.cx)
            else:
                hx, cx = state
            hx = hx if hx is None else hx.expand(latent_rgb.size(0), -1)
            cx = cx if cx is None else cx.expand(latent_rgb.size(0), -1)
            state_prev = (hx, cx)
            encoder_out = self.belief_encoder(latent_rgb, latent_depth, *state_prev)
            res["encoder_out"] = encoder_out
            state_new = encoder_out["hx"], encoder_out["cx"]

            belief_state = encoder_out["belief_state"]
            extracted_obs = torch.cat([latent_rgb, belief_state], dim=1)

            if self.use_belief_decoder:
                decoder_out = self.belief_decoder(hx, latent_depth)
                res["decoder_out"] = decoder_out
                if self.use_priv_decoder:
                    res["priv_decoded"] = decoder_out["priv_decoded"]
        else:
            extracted_obs = torch.cat([latent_rgb, latent_depth], dim=1)

        t_in = extracted_obs
        rot_in = extracted_obs
        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_t_dim = self.hidden_dim if self.use_mlp_for_prev_pose else self.t_mlp_out_dim
                prev_rot_dim = self.hidden_dim if self.use_mlp_for_prev_pose else self.rot_mlp_out_dim
                prev_pose = {
                    "t": torch.zeros(latent_rgb.size(0), prev_t_dim, device=latent_rgb.device),
                    "rot": torch.zeros(latent_rgb.size(0), prev_rot_dim, device=latent_rgb.device),
                }
            if self.do_predict_2d_t:
                prev_pose["center_depth"] = torch.zeros(
                    latent_rgb.size(0), self.depth_mlp_out_dim, device=latent_rgb.device
                )
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
                prev_latent = torch.zeros(latent_rgb.size(0), self.depth_dim + self.rgb_dim, device=latent_rgb.device)
            t_in = torch.cat([t_in, prev_latent], dim=1)
            rot_in = torch.cat([rot_in, prev_latent], dim=1)

        t = self.t_mlp(t_in)
        res.update(
            {
                "latent_depth": latent_depth,
                "state": state_new,
                "t": t,
            }
        )

        if self.use_prev_latent:
            res["prev_latent"] = extracted_obs

        if self.do_predict_rot:
            rot = self.rot_mlp(rot_in)
            res["rot"] = rot

        if self.do_predict_2d_t:
            depth_in = extracted_obs
            if self.use_prev_pose_condition:
                depth_in = torch.cat([depth_in, prev_pose["center_depth"]], dim=1)
            if self.use_prev_latent:
                depth_in = torch.cat([depth_in, prev_latent], dim=1)
            center_depth = self.depth_mlp(depth_in)
            res["center_depth"] = center_depth

        if self.do_predict_kpts:
            kpts = self.kpts_mlp(extracted_obs)
            kpts = kpts.view(-1, 8 + 24, 2)
            res["kpts"] = kpts

        return res


class RecurrentCNNSeparated(RecurrentCNN):
    """ """

    def __init__(
        self,
        *args,
        roi_size=7,
        **kwargs,
    ):
        use_obs_belief = kwargs.pop("use_obs_belief", False)
        kwargs["use_obs_belief"] = False
        super().__init__(*args, **kwargs)

        self.t_rnn_kwargs = copy.deepcopy(kwargs)
        self.t_rnn_kwargs["rt_mlps_num_layers"] = 1
        self.t_rnn_kwargs["encoder_name"] = None
        self.t_rnn_kwargs["do_predict_rot"] = False
        self.t_rnn_kwargs["use_obs_belief"] = use_obs_belief
        self.t_rnn = RecurrentCNN(
            *args,
            **self.t_rnn_kwargs,
        )

        if self.encoder_name == "resnet50":
            self.mid_feature_dim = 1024
            self.mid_layer_name = "layer3"
        elif self.encoder_name == "resnet18":
            self.mid_feature_dim = 256
            self.mid_layer_name = "layer3"
        elif self.encoder_name == "regnet_y_800mf":
            self.mid_feature_dim = 320
            self.mid_layer_name = "trunk_output.block3"
        else:
            raise ValueError(f"Unknown encoder_name: {self.encoder_name}")

        self.roi_size = roi_size
        self.rot_mlp_in_dim = self.mid_feature_dim * 2
        self.rot_mlp_out_dim = 6 if self.do_predict_6d_rot else (3 if self.do_predict_3d_rot else 4)

        if self.use_prev_pose_condition:
            self.rot_mlp_in_dim += self.rot_mlp_out_dim

        self.rot_mlp = MLP(
            in_dim=self.rot_mlp_in_dim,
            out_dim=self.rot_mlp_out_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.rt_mlps_num_layers,  # rot head should capture rot info
            dropout=self.dropout,
        )
        self.t_mlp = None

        self.hx = None
        self.cx = None

        self.intermediate_outputs = {}

        def hook_function(module, input, output):
            self.intermediate_outputs[self.mid_layer_name] = output

        # Register the hook
        getattr(self.encoder_img, self.mid_layer_name).register_forward_hook(hook_function)
        getattr(self.encoder_depth, self.mid_layer_name).register_forward_hook(hook_function)

        self.rgb_roi_cnn = nn.Sequential(
            nn.Conv2d(self.mid_feature_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.depth_roi_cnn = nn.Sequential(
            nn.Conv2d(self.mid_feature_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def reset_state(self, batch_size, device):
        self.t_rnn.reset_state(batch_size, device)
        super().reset_state(batch_size, device)

    def detach_state(self):
        super().detach_state()
        self.t_rnn.detach_state()

    def forward(
        self, rgb, depth, prev_pose=None, latent_rgb=None, latent_depth=None, prev_latent=None, state=None, **kwargs
    ):

        bbox = kwargs["bbox"]
        B, C, H, W = rgb.size()

        latent_rgb = self.encoder_img(rgb)
        latent_depth = self.encoder_depth(depth)

        mid_layer_output_rgb = self.intermediate_outputs[self.mid_layer_name]
        mid_layer_output_depth = self.intermediate_outputs[self.mid_layer_name]
        if isinstance(bbox, torch.Tensor):
            ind = torch.arange(bbox.shape[0]).unsqueeze(1)
            ind = ind.type_as(bbox)
            bbox_roi = torch.cat((ind, bbox.reshape(-1, 4).float()), dim=1).float()
        else:
            bbox_roi = bbox

        latent_rgb_roi = roi_align(mid_layer_output_rgb, bbox_roi, self.roi_size)
        latent_depth_roi = roi_align(mid_layer_output_depth, bbox_roi, self.roi_size)
        latent_rgb_roi = self.rgb_roi_cnn(latent_rgb_roi).view(B, -1)
        latent_depth_roi = self.depth_roi_cnn(latent_depth_roi).view(B, -1)

        t_net_out = self.t_rnn(
            rgb,
            depth,
            prev_pose=prev_pose,
            latent_rgb=latent_rgb,
            latent_depth=latent_depth,
            prev_latent=prev_latent,
            state=state,
        )

        extracted_obs_rot = torch.cat([latent_rgb_roi, latent_depth_roi], dim=1)

        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_pose = {
                    "rot": torch.zeros(latent_rgb_roi.size(0), self.rot_mlp_out_dim, device=latent_rgb_roi.device),
                }
            rot_in = torch.cat([extracted_obs_rot, prev_pose["rot"]], dim=1)
        else:
            rot_in = extracted_obs_rot

        rot = self.rot_mlp(rot_in.view(B, -1))

        res = {}
        res.update(
            {
                "latent_depth_roi": latent_depth_roi,
                "latent_depth": latent_depth,
                "state": t_net_out["state"],
                "t": t_net_out["t"],
                "rot": rot,
                "decoder_out": t_net_out.get("decoder_out"),
            }
        )

        if self.use_prev_latent:
            res["prev_latent"] = t_net_out["prev_latent"]

        if self.do_predict_2d_t:
            res["center_depth"] = t_net_out["center_depth"]

        if self.do_predict_kpts:
            kpts = self.kpts_mlp(t_net_out["prev_latent"])
            kpts = kpts.view(-1, 8 + 24, 2)
            res["kpts"] = kpts

        return res
