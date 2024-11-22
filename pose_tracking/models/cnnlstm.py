import torch
import torch.nn as nn
from pose_tracking.models.encoders import get_encoders
from pose_tracking.models.rnn_cells import GRUCell
from torch import jit
from torch.nn import Parameter


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input, state):
        hx, cx = state
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return {"hidden_state": hy, "cell_state": cy}


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
        self.hidden_attn = nn.Sequential(
            MLP(
                in_dim=state_dim,
                out_dim=depth_decoder_out_dim,
                hidden_dim=hidden_attn_hidden_dim,
                num_layers=hidden_attn_num_layers,
                dropout=dropout,
            ),
            nn.Sigmoid(),
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


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1, act="gelu", act_out=None, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        if num_layers > 1:
            self.layers = [nn.Linear(in_dim, hidden_dim)]
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            for i in range(num_layers - 2):
                self.layers = [nn.Linear(hidden_dim, hidden_dim)]
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            self.layers = [nn.Linear(in_dim, out_dim)]
        self.layers = nn.ModuleList(self.layers)
        self.act = nn.GELU() if act == "gelu" else nn.LeakyReLU()
        self.act_out = act_out

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        if self.act_out is not None:
            x = self.act_out(x)
        return x


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
        dropout=0.0,
        rnn_type="gru",
        encoder_name="regnet_y_800mf",
        do_predict_2d_t=False,
        do_predict_6d_rot=False,
        use_rnn=True,
        use_obs_belief=False,
        use_priv_decoder=False,
        do_freeze_encoders=False,
        use_prev_pose_condition=False,
        do_predict_kpts=False,
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
        self.benc_belief_enc_num_layers = benc_belief_enc_num_layers
        self.benc_belief_depth_enc_num_layers = benc_belief_depth_enc_num_layers
        self.rnn_type = rnn_type
        self.encoder_name = encoder_name
        self.dropout = dropout
        self.do_predict_2d_t = do_predict_2d_t
        self.do_predict_6d_rot = do_predict_6d_rot
        self.use_obs_belief = use_obs_belief
        self.use_priv_decoder = use_priv_decoder
        self.use_rnn = use_rnn
        self.do_freeze_encoders = do_freeze_encoders
        self.use_prev_pose_condition = use_prev_pose_condition
        self.do_predict_kpts = do_predict_kpts

        self.input_dim = depth_dim + rgb_dim

        if use_rnn:
            if rnn_type == "lstm":
                self.state_cell = nn.LSTMCell(self.input_dim, hidden_dim)
            elif rnn_type == "lstm_custom":
                self.state_cell = LSTMCell(self.input_dim, hidden_dim)
            elif rnn_type == "gru":
                self.state_cell = nn.GRUCell(self.input_dim, hidden_dim)
            elif rnn_type == "gru_custom":
                self.state_cell = GRUCell(self.input_dim, hidden_dim)
            else:
                raise ValueError("rnn_type must be 'lstm' or 'gru'")
        else:
            self.state_cell = MLP(
                in_dim=self.input_dim,
                out_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_layers=1,
                dropout=dropout,
            )

        if use_obs_belief:
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
        if do_predict_2d_t:
            self.t_mlp_out_dim = 2
            self.depth_mlp_in_dim = depth_dim + rgb_dim
            self.depth_mlp_out_dim = 1
            if use_prev_pose_condition:
                self.depth_mlp_in_dim += self.depth_mlp_out_dim
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
        self.rot_mlp_out_dim = 6 if do_predict_6d_rot else 4

        if use_prev_pose_condition:
            self.t_mlp_in_dim += self.t_mlp_out_dim
            self.rot_mlp_in_dim += self.rot_mlp_out_dim

        self.t_mlp = MLP(
            in_dim=self.t_mlp_in_dim,
            out_dim=self.t_mlp_out_dim,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            act_out=nn.Sigmoid() if do_predict_2d_t else None,  # normalized coords
            dropout=dropout,
        )
        self.rot_mlp = MLP(
            in_dim=self.rot_mlp_in_dim,
            out_dim=self.rot_mlp_out_dim,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            dropout=dropout,
        )

        if self.do_predict_kpts:
            self.kpts_mlp_in_dim = depth_dim + rgb_dim
            self.kpts_mlp_out_dim = (8 + 24) * 2
            self.kpts_mlp = MLP(
                in_dim=self.kpts_mlp_in_dim,
                out_dim=self.kpts_mlp_out_dim,
                hidden_dim=hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout,
            )

        self.encoder_name = encoder_name
        self.encoder_img, self.encoder_depth = get_encoders(encoder_name, do_freeze=do_freeze_encoders)
        self.hx = None
        self.cx = None

    def reset_state(self, batch_size, device):
        # should be called at the beginning of each sequence
        self.hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.cx = None if "gru" in self.rnn_type else torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, rgb, depth, prev_pose=None):

        latent_rgb = self.encoder_img(rgb)
        latent_depth = self.encoder_depth(depth)

        res = {}
        if self.use_obs_belief:
            encoder_out = self.belief_encoder(latent_rgb, latent_depth, self.hx, self.cx)
            self.hx, self.cx = encoder_out["hx"], encoder_out["cx"]
            decoder_out = self.belief_decoder(self.hx, latent_depth)

            belief_state = encoder_out["belief_state"]
            extracted_obs = torch.cat([latent_rgb, belief_state], dim=1)

            res.update(
                {
                    "encoder_out": encoder_out,
                    "decoder_out": decoder_out,
                }
            )
            if self.use_priv_decoder:
                res["priv_decoded"] = decoder_out["priv_decoded"]
        else:
            extracted_obs = torch.cat([latent_rgb, latent_depth], dim=1)

        if self.use_prev_pose_condition:
            if prev_pose is None:
                prev_pose = {
                    "t": torch.zeros(latent_rgb.size(0), self.t_mlp_out_dim, device=latent_rgb.device),
                    "rot": torch.zeros(latent_rgb.size(0), self.rot_mlp_out_dim, device=latent_rgb.device),
                }
            if self.do_predict_2d_t:
                prev_pose["center_depth"] = torch.zeros(
                    latent_rgb.size(0), self.depth_mlp_out_dim, device=latent_rgb.device
                )
            t_in = torch.cat([extracted_obs, prev_pose["t"]], dim=1)
            rot_in = torch.cat([extracted_obs, prev_pose["rot"]], dim=1)
        else:
            t_in = extracted_obs
            rot_in = extracted_obs

        t = self.t_mlp(t_in)
        rot = self.rot_mlp(rot_in)

        res.update(
            {
                "latent_depth": latent_depth,
                "state": {"hx": self.hx, "cx": self.cx},
                "t": t,
                "rot": rot,
            }
        )

        if self.do_predict_2d_t:
            if self.use_prev_pose_condition:
                depth_in = torch.cat([extracted_obs, prev_pose["center_depth"]], dim=1)
            else:
                depth_in = extracted_obs
            center_depth = self.depth_mlp(depth_in)
            res["center_depth"] = center_depth

        if self.do_predict_kpts:
            kpts = self.kpts_mlp(extracted_obs)
            kpts = kpts.view(-1, 8 + 24, 2)
            res["kpts"] = kpts

        return res
