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
        state_dim,
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
            in_dim=state_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_enc_hidden_dim,
            num_layers=belief_enc_num_layers,
            dropout=dropout,
        )
        self.belief_depth_mlp = MLP(
            in_dim=state_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_depth_enc_hidden_dim,
            num_layers=belief_depth_enc_num_layers,
            dropout=dropout,
        )

    def forward(self, latent_rgb, latent_depth, state):
        latent_obs = torch.cat([latent_rgb, latent_depth], dim=-1)
        if self.use_rnn:
            cell_out = self.state_cell(latent_obs, state)
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
            self.droputs = []
            self.layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim)))
            for i in range(num_layers - 2):
                if dropout > 0:
                    self.droputs.append(nn.Dropout(dropout))
                self.layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)))
            if dropout > 0:
                self.droputs.append(nn.Dropout(dropout))
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
        for layer_idx, layer in enumerate(self.layers[:-1]):
            x = self.act(layer(x))
            if layer_idx < len(self.droputs):
                x = self.droputs[layer_idx](x)
        x = self.layers[-1](x)
        if self.act_out is not None:
            x = self.act_out(x)
        return x

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())


class RecurrentNet(nn.Module):
    def __init__(self, rnn_type="gru", rnn_state_init_type="zeros", state_dim=256):
        super().__init__()

        self.rnn_state_init_type = rnn_state_init_type
        self.state_dim = state_dim
        self.rnn_type = rnn_type

        if self.rnn_state_init_type == "learned":
            self.hx = nn.Parameter(torch.randn(1, self.state_dim, device="cpu"))
            self.cx = None if "gru" in self.rnn_type else nn.Parameter(torch.randn(1, self.state_dim, device="cpu"))
        else:
            self.hx = None
            self.cx = None

    def __repr__(self):
        return print_cls(self, extra_str=super().__repr__())

    def reset_state(self, batch_size, device):
        if self.rnn_state_init_type == "learned":
            self.detach_state()
        elif self.rnn_state_init_type == "zeros":
            self.hx = torch.zeros(1, self.state_dim, device=device)
            self.cx = None if "gru" in self.rnn_type else torch.zeros(1, self.state_dim, device=device)
        elif self.rnn_state_init_type == "rand":
            self.hx = torch.randn(1, self.state_dim, device=device)
            self.cx = None if "gru" in self.rnn_type else torch.randn(1, self.state_dim, device=device)
        else:
            raise ValueError(f"Unknown rnn_state_init_type: {self.rnn_state_init_type}")

    def detach_state(self):
        if self.training:
            self.hx.detach_()
            if self.cx is not None:
                self.cx.detach_()


class RecurrentCNNVanilla(RecurrentNet):
    def __init__(
        self,
        depth_dim,
        rgb_dim,
        hidden_dim=256,
        state_dim=512,
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
        use_mlp_for_prev_pose=False,
        do_freeze_encoders=False,
        use_prev_pose_condition=False,
        use_prev_latent=False,
        do_predict_kpts=False,
        do_predict_rot=True,
        do_predict_t=True,
        do_predict_abs_pose=False,
        use_kpts_for_rot=False,
        use_depth=True,
        r_num_layers_inc=0,
        rt_hidden_dim=None,
        extracted_obs_dim=None,
        bbox_num_kpts=8 + 24,
    ):
        super().__init__(rnn_type=rnn_type, rnn_state_init_type=rnn_state_init_type, state_dim=state_dim)

        self.depth_dim = depth_dim
        self.rgb_dim = rgb_dim
        self.hidden_dim = hidden_dim
        self.encoder_out_dim = encoder_out_dim
        self.rnn_type = rnn_type
        self.encoder_name = encoder_name
        self.dropout = dropout
        self.dropout_heads = dropout_heads
        self.rt_mlps_num_layers = rt_mlps_num_layers
        self.rnn_state_init_type = rnn_state_init_type
        self.encoder_img_weights = encoder_img_weights
        self.encoder_depth_weights = encoder_depth_weights
        self.norm_layer_type = norm_layer_type

        self.use_depth = use_depth
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.do_predict_3d_rot = do_predict_3d_rot
        self.do_predict_abs_pose = do_predict_abs_pose
        self.use_rnn = use_rnn
        self.do_freeze_encoders = do_freeze_encoders
        self.use_prev_pose_condition = use_prev_pose_condition
        self.use_prev_latent = use_prev_latent
        self.do_predict_kpts = do_predict_kpts
        self.do_predict_rot = do_predict_rot
        self.do_predict_t = do_predict_t
        self.use_mlp_for_prev_pose = use_mlp_for_prev_pose
        self.use_kpts_for_rot = use_kpts_for_rot

        self.input_dim = depth_dim + rgb_dim if use_depth else rgb_dim
        self.rt_hidden_dim = hidden_dim // 2 if rt_hidden_dim is None else rt_hidden_dim
        self.extracted_obs_dim = state_dim if extracted_obs_dim is None else extracted_obs_dim
        self.num_kpts = bbox_num_kpts

        if use_rnn:
            if rnn_type == "lstm":
                self.state_cell = nn.LSTMCell(self.input_dim, state_dim)
            else:
                self.state_cell = nn.GRUCell(self.input_dim, state_dim)

            for name, param in self.state_cell.named_parameters():
                if "weight" in name:
                    nn.init.xavier_normal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
            self.state_cell.bias_hh.data.fill_(0.5)  # memorize last state
        else:
            self.state_cell = MLP(
                in_dim=self.input_dim,
                out_dim=self.state_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
            )
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = depth_dim + rgb_dim if use_depth else rgb_dim
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

        self.t_mlp_in_dim = self.rot_mlp_in_dim = self.extracted_obs_dim
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else (3 if do_predict_3d_rot else 4)

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
                num_layers=rt_mlps_num_layers + r_num_layers_inc,
                dropout=dropout_heads,
            )
        if do_predict_abs_pose:
            self.t_mlp_abs_pose = MLP(
                in_dim=self.t_mlp_in_dim,
                out_dim=self.t_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                act_out=nn.Sigmoid() if do_predict_2d_t else None,  # normalized coords
                dropout=dropout_heads,
            )
            self.rot_mlp_abs_pose = MLP(
                in_dim=self.rot_mlp_in_dim,
                out_dim=self.rot_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers + r_num_layers_inc,
                dropout=dropout_heads,
            )

        if self.do_predict_kpts:
            self.kpts_mlp_in_dim = self.extracted_obs_dim
            if self.use_prev_latent:
                self.kpts_mlp_in_dim += self.input_dim
            self.kpts_mlp_out_dim = self.num_kpts * 2
            self.kpts_mlp = MLP(
                in_dim=self.kpts_mlp_in_dim,
                out_dim=self.kpts_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout_heads,
            )

        if use_kpts_for_rot:
            self.rot_mlp_in_dim = self.extracted_obs_dim + self.kpts_mlp_out_dim
            self.rot_mlp = MLP(
                in_dim=self.rot_mlp_in_dim,
                out_dim=self.rot_mlp_out_dim,
                hidden_dim=self.rt_hidden_dim,
                num_layers=rt_mlps_num_layers + r_num_layers_inc,
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
            if not use_depth:
                self.encoder_depth = None

    def forward(
        self,
        rgb,
        depth,
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
        state_prev = self.prep_state(state, bs)
        res = self.extract_img_info(latent_rgb, latent_depth, state_prev)
        extracted_obs = res["extracted_obs"]
        latent = res["latent"]

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

        if self.do_predict_t:
            t = self.t_mlp(t_in)
        else:
            t = torch.zeros(bs, self.t_mlp_out_dim, device=rgb.device)
        res["t"] = t

        if self.do_predict_kpts:
            kpt_in = extracted_obs
            if self.use_prev_latent:
                kpt_in = torch.cat([kpt_in, prev_latent], dim=1)
            kpts = self.kpts_mlp(kpt_in).sigmoid()
            kpts = kpts.view(bs, self.num_kpts, 2)
            res["kpts"] = kpts

        if self.do_predict_rot:
            if self.use_kpts_for_rot:
                rot = self.rot_mlp(torch.cat([extracted_obs, kpts.view(bs, -1)], dim=1))
            else:
                rot = self.rot_mlp(rot_in)
            res["rot"] = rot
        else:
            rot = torch.zeros(bs, self.rot_mlp_out_dim, device=rgb.device)
        res["rot"] = rot

        if self.do_predict_abs_pose:
            t_abs_pose = self.t_mlp_abs_pose(t_in)
            rot_abs_pose = self.rot_mlp_abs_pose(rot_in)
            res.update(
                {
                    "t_abs_pose": t_abs_pose,
                    "rot_abs_pose": rot_abs_pose,
                }
            )

        if self.do_predict_2d_t:
            depth_in = extracted_obs
            if self.use_prev_pose_condition:
                depth_in = torch.cat([depth_in, prev_pose["t"][:, 2]], dim=1)
            if self.use_prev_latent:
                depth_in = torch.cat([depth_in, prev_latent], dim=1)
            center_depth = self.depth_mlp(depth_in)
            res["center_depth"] = center_depth

        res["latent_depth"] = latent_depth
        return res

    def extract_img_info(self, latent_rgb, latent_depth, state_prev):

        if self.use_depth:
            latent = torch.cat([latent_rgb, latent_depth], dim=1)
        else:
            latent = latent_rgb
        state_new = self.state_cell(latent, state_prev)

        extracted_obs = self.postp_state(state_new)

        res = {}
        res["state"] = state_new
        res["extracted_obs"] = extracted_obs
        res["latent"] = latent

        return res

    def postp_state(self, state_new):
        return state_new[0] if self.rnn_type == "lstm" else state_new

    def prep_state(self, state, bs):
        if state is None:
            hx, cx = (self.hx, self.cx)
        else:
            if self.rnn_type == "lstm":
                hx, cx = state
            else:
                hx = state[0] if type(state) in (tuple, list) else state
                cx = None
        hx = hx if hx is None else hx.expand(bs, -1)
        cx = cx if cx is None else cx.expand(bs, -1)
        state_prev = (hx, cx) if self.rnn_type == "lstm" else hx
        return state_prev


class RecurrentCNN(RecurrentCNNVanilla):
    def __init__(
        self,
        *args,
        bdec_priv_decoder_out_dim=None,
        belief_hidden_dim=256,
        belief_num_layers=2,
        use_obs_belief=False,
        use_priv_decoder=False,
        use_belief_decoder=False,
        **kwargs,
    ):

        self.use_obs_belief = use_obs_belief
        self.use_belief_decoder = use_belief_decoder
        self.use_priv_decoder = use_priv_decoder
        self.use_depth = kwargs["use_depth"]

        self.belief_hidden_dim = belief_hidden_dim
        self.rgb_dim = kwargs["rgb_dim"]
        self.depth_dim = kwargs["depth_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.state_dim = kwargs["state_dim"]

        if use_obs_belief:
            extracted_obs_dim = self.rgb_dim + self.depth_dim
        else:
            extracted_obs_dim = self.state_dim

        super().__init__(
            *args,
            **kwargs,
            extracted_obs_dim=extracted_obs_dim,
        )

        if self.use_obs_belief:
            self.belief_encoder = BeliefEncoder(
                state_cell=self.state_cell,
                state_dim=self.state_dim,
                depth_latent_dim=self.depth_dim,
                belief_enc_hidden_dim=belief_hidden_dim,
                belief_depth_enc_hidden_dim=belief_hidden_dim,
                belief_enc_num_layers=belief_num_layers,
                belief_depth_enc_num_layers=belief_num_layers,
                use_rnn=self.use_rnn,
                dropout=self.dropout,
            )
            if use_belief_decoder:
                assert bdec_priv_decoder_out_dim
                self.belief_decoder = BeliefDecoder(
                    state_dim=self.state_dim,
                    priv_decoder_out_dim=bdec_priv_decoder_out_dim,
                    priv_decoder_hidden_dim=belief_hidden_dim,
                    depth_decoder_out_dim=self.depth_dim,
                    depth_decoder_hidden_dim=belief_hidden_dim,
                    hidden_attn_hidden_dim=belief_hidden_dim,
                    priv_decoder_num_layers=belief_num_layers,
                    depth_decoder_num_layers=belief_num_layers,
                    hidden_attn_num_layers=belief_num_layers,
                    use_priv_decoder=use_priv_decoder,
                    dropout=self.dropout,
                )
            else:
                self.belief_decoder = None

    def extract_img_info(self, latent_rgb, latent_depth, state_prev):
        res = {}

        if self.use_obs_belief:
            encoder_out = self.belief_encoder(latent_rgb, latent_depth, state_prev)
            res["encoder_out"] = encoder_out
            hx = encoder_out["hx"]
            state_new = hx, encoder_out["cx"]
            latent_depth_post = encoder_out["latent_depth_gated"]
            posterior_belief = encoder_out["posterior_belief"]
            extracted_obs = torch.cat([latent_rgb, posterior_belief], dim=1)

            if self.use_belief_decoder:
                decoder_out = self.belief_decoder(hx, latent_depth)
                latent_depth_post = decoder_out["depth_final"]
                res["decoder_out"] = decoder_out
                if self.use_priv_decoder:
                    res["priv_decoded"] = decoder_out["priv_decoded"]
            latent = torch.cat([latent_rgb, latent_depth_post], dim=1)
        else:
            if self.use_depth:
                latent = torch.cat([latent_rgb, latent_depth], dim=1)
            else:
                latent = latent_rgb
            if self.use_rnn:
                state_new = self.state_cell(latent, state_prev)
            else:
                state_new = self.state_cell(latent)
            state_new_postp = self.postp_state(state_new)
            extracted_obs = state_new_postp

        res["state"] = state_new
        res["latent"] = latent
        res["extracted_obs"] = extracted_obs
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
            dropout=self.dropout_heads,
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

    def detach_state(self):
        self.t_rnn.detach_state()

    def forward(
        self, rgb, depth, prev_pose=None, latent_rgb=None, latent_depth=None, prev_latent=None, state=None, **kwargs
    ):

        bbox = kwargs["bbox"]
        bs, C, H, W = rgb.size()

        latent_rgb = self.encoder_img(rgb)
        latent_depth = self.encoder_depth(depth)

        mid_layer_output_rgb = self.intermediate_outputs[self.mid_layer_name]
        mid_layer_output_depth = self.intermediate_outputs[self.mid_layer_name]
        if isinstance(bbox, torch.Tensor):
            ind = torch.arange(bbox.shape[0]).unsqueeze(1)
            ind = ind.type_as(bbox)
            bbox_roi = torch.cat((ind, bbox.reshape(-1, 4)), dim=1).float()
        else:
            bbox_roi = bbox

        latent_rgb_roi = roi_align(mid_layer_output_rgb, bbox_roi, self.roi_size)
        latent_depth_roi = roi_align(mid_layer_output_depth, bbox_roi, self.roi_size)
        latent_rgb_roi = self.rgb_roi_cnn(latent_rgb_roi).view(bs, -1)
        latent_depth_roi = self.depth_roi_cnn(latent_depth_roi).view(bs, -1)

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

        rot = self.rot_mlp(rot_in.view(bs, -1))

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
            res["latent"] = t_net_out["latent"]

        if self.do_predict_2d_t:
            res["center_depth"] = t_net_out["center_depth"]

        if self.do_predict_kpts:
            kpts = self.kpts_mlp(t_net_out["latent"])
            kpts = kpts.view(bs, 8 + 24, 2)
            res["kpts"] = kpts

        return res


class RecurrentCNNDouble(RecurrentCNN):
    """ """

    def __init__(
        self,
        *args,
        use_crop_for_rot=True,
        **kwargs,
    ):
        self.use_crop_for_rot = use_crop_for_rot

        use_obs_belief = kwargs.pop("use_obs_belief", False)
        use_belief_decoder = kwargs.pop("use_belief_decoder", False)
        kwargs["use_obs_belief"] = False
        kwargs["use_belief_decoder"] = False
        super().__init__(*args, **kwargs)

        self.t_rnn_kwargs = copy.deepcopy(kwargs)
        self.t_rnn_kwargs["encoder_name"] = None
        self.t_rnn_kwargs["do_predict_rot"] = False
        self.t_rnn_kwargs["use_obs_belief"] = use_obs_belief
        self.t_rnn_kwargs["use_belief_decoder"] = use_belief_decoder
        self.rot_rnn_kwargs = copy.deepcopy(self.t_rnn_kwargs)
        self.rot_rnn_kwargs["do_predict_t"] = False
        self.rot_rnn_kwargs["do_predict_rot"] = self.do_predict_rot
        if kwargs["do_predict_t"]:
            self.t_rnn = RecurrentCNN(
                *args,
                **self.t_rnn_kwargs,
            )
        if kwargs["do_predict_rot"]:
            self.rot_rnn = RecurrentCNN(
                *args,
                **self.rot_rnn_kwargs,
            )
            if use_crop_for_rot:
                self.encoder_img_rot, self.encoder_depth_rot = get_encoders(
                    self.encoder_name,
                    do_freeze=self.do_freeze_encoders,
                    weights_rgb=self.encoder_img_weights,
                    weights_depth=self.encoder_depth_weights,
                    norm_layer_type=self.norm_layer_type,
                    out_dim=self.encoder_out_dim,
                )
                if not self.use_depth:
                    self.encoder_depth_rot = None
        if self.do_predict_rot and self.use_crop_for_rot and not self.do_predict_t:
            self.encoder_img = None

        self.rot_mlp = None
        self.t_mlp = None
        self.state_cell = None
        self.prev_t_mlp = None
        self.prev_rot_mlp = None

    def reset_state(self, batch_size, device):
        if self.do_predict_t:
            self.t_rnn.reset_state(batch_size, device)
        if self.do_predict_rot:
            self.rot_rnn.reset_state(batch_size, device)

    def detach_state(self):
        if self.do_predict_t:
            self.t_rnn.detach_state()
        if self.do_predict_rot:
            self.rot_rnn.detach_state()

    def forward(
        self, rgb, depth, prev_pose=None, latent_rgb=None, latent_depth=None, prev_latent=None, state=None, **kwargs
    ):

        bbox = kwargs["bbox"]
        bs, C, H, W = rgb.size()

        if self.do_predict_rot and self.use_crop_for_rot and not self.do_predict_t:
            latent_rgb = None
        else:
            latent_rgb = self.encoder_img(rgb)
        if self.use_depth:
            latent_depth = self.encoder_depth(depth)
        else:
            latent_depth = None

        if state is not None:
            if self.do_predict_rot and self.do_predict_t:
                state_t, state_rot = state
            elif self.do_predict_t:
                state_t = state[0]
            else:
                state_rot = state[0]
        else:
            state_t = state_rot = None

        if prev_latent is not None:
            if self.do_predict_rot and self.do_predict_t:
                prev_latent_t = prev_latent[:, : self.encoder_out_dim * 2]
                prev_latent_rot = prev_latent[:, self.encoder_out_dim * 2 :]
            elif self.do_predict_t:
                prev_latent_t = prev_latent[:, : self.encoder_out_dim * 2]
            else:
                prev_latent_rot = prev_latent[:, : self.encoder_out_dim * 2]
        else:
            prev_latent_t = prev_latent_rot = None

        res = {}
        state = []
        decoder_out = []
        latent = []
        if self.do_predict_t:
            t_net_out = self.t_rnn(
                rgb,
                depth,
                prev_pose=prev_pose,
                latent_rgb=latent_rgb,
                latent_depth=latent_depth,
                prev_latent=prev_latent_t,
                state=state_t,
            )
            res["t"] = t_net_out["t"]
            state += [t_net_out["state"]]
            decoder_out += [t_net_out.get("decoder_out")]
            latent += [t_net_out["latent"]]
        else:
            res["t"] = torch.zeros(bs, self.t_mlp_out_dim, device=rgb.device)

        if self.do_predict_rot:
            if self.use_crop_for_rot:
                padding = 5
                new_boxes = []
                for i, boxes_padded in enumerate(bbox):
                    boxes_padded = boxes_padded.clone()
                    boxes_padded[..., 0] = boxes_padded[..., 0] - padding
                    boxes_padded[..., 1] = boxes_padded[..., 1] - padding
                    boxes_padded[..., 2] = boxes_padded[..., 2] + padding
                    boxes_padded[..., 3] = boxes_padded[..., 3] + padding
                    image_size = rgb.shape[-2:]
                    H, W = image_size
                    boxes_padded[..., 0].clamp_(min=0, max=W)
                    boxes_padded[..., 1].clamp_(min=0, max=H)
                    boxes_padded[..., 2].clamp_(min=0, max=W)
                    boxes_padded[..., 3].clamp_(min=0, max=H)
                    new_boxes.append(boxes_padded)
                crop_size = (60, 80)
                crop_size = (60 * 2, 80 * 2)
                rgb_crop = roi_align(rgb, new_boxes, crop_size)
                depth_crop = roi_align(depth, new_boxes, crop_size)
                latent_rgb_rot = self.encoder_img_rot(rgb_crop)
                latent_depth_rot = self.encoder_depth_rot(depth_crop) if self.use_depth else None
            else:
                latent_rgb_rot = latent_rgb
                latent_depth_rot = latent_depth

            rot_net_out = self.rot_rnn(
                rgb,
                depth,
                prev_pose=prev_pose,
                latent_rgb=latent_rgb_rot,
                latent_depth=latent_depth_rot,
                prev_latent=prev_latent_rot,
                state=state_rot,
            )
            res["rot"] = rot_net_out["rot"]
            state += [rot_net_out["state"]]
            decoder_out += [rot_net_out.get("decoder_out")]
            latent += [rot_net_out["latent"]]
        else:
            res["rot"] = torch.zeros(bs, self.rot_mlp_out_dim, device=rgb.device)

        res.update(
            {
                "latent_depth": latent_depth,
                "state": state,
                "decoder_out": (torch.cat(decoder_out, dim=1) if self.use_belief_decoder else None),
            }
        )

        if self.use_prev_latent:
            res["latent"] = torch.cat(latent, dim=1)

        if self.do_predict_2d_t:
            res["center_depth"] = t_net_out["center_depth"]

        if self.do_predict_kpts:
            kpts = self.kpts_mlp(torch.cat(latent, dim=1))
            kpts = kpts.view(bs, 8 + 24, 2)
            res["kpts"] = kpts

        return res
