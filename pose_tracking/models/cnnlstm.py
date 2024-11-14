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
    def forward(self, input, hx, cx):
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class StateMLP(nn.Module):
    # maps vector1 to vector2
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.state_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.layers = [nn.Linear(in_dim, hidden_dim)]
        for i in range(num_layers - 2):
            self.layers = [nn.Linear(hidden_dim, hidden_dim)]
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.ModuleList(self.layers)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        return x


class BeliefEncoder(nn.Module):
    def __init__(
        self,
        rnn_cell,
        rnn_hidden_dim,
        depth_latent_dim,
        belief_enc_hidden_dim,
        belief_depth_enc_hidden_dim,
        belief_enc_num_layers=2,
        belief_depth_enc_num_layers=2,
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.depth_latent_dim = depth_latent_dim

        self.belief_prior_mlp = StateMLP(
            in_dim=rnn_hidden_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_enc_hidden_dim,
            num_layers=belief_enc_num_layers,
        )
        self.belief_depth_mlp = StateMLP(
            in_dim=rnn_hidden_dim,
            out_dim=depth_latent_dim,
            hidden_dim=belief_depth_enc_hidden_dim,
            num_layers=belief_depth_enc_num_layers,
        )

    def forward(self, latent_rgb, latent_depth, hx, cx=None):
        latent_obs = torch.cat([latent_rgb, latent_depth], dim=-1)
        if cx is None:
            cell_out = self.rnn_cell(latent_obs, hx)
            cx_new = None
        else:
            cell_out = self.rnn_cell(latent_obs, hx, cx)
            cx_new = cell_out["cell_state"]
        prior_belief = cell_out["hidden_state"] if isinstance(cell_out, dict) else cell_out
        prior_belief_encoded = self.belief_prior_mlp(prior_belief)
        prior_belief_depth_encoded = self.belief_depth_mlp(prior_belief)
        latent_depth_gated = latent_depth * torch.sigmoid(prior_belief_depth_encoded)
        posterior_belief = prior_belief_encoded + latent_depth_gated
        return {
            "posterior_belief": posterior_belief,
            "prior_belief": prior_belief,
            "hidden_state": prior_belief,
            "belief_state": posterior_belief,
            "prior_belief_encoded": prior_belief_encoded,
            "prior_belief_depth_encoded": prior_belief_depth_encoded,
            "latent_depth_gated": latent_depth_gated,
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
    ):
        super().__init__()
        self.hidden_attn = nn.Sequential(
            StateMLP(
                in_dim=state_dim,
                out_dim=depth_decoder_out_dim,
                hidden_dim=hidden_attn_hidden_dim,
                num_layers=hidden_attn_num_layers,
            ),
            nn.Sigmoid(),
        )
        self.depth_decoder = StateMLP(
            in_dim=state_dim,
            out_dim=depth_decoder_out_dim,
            hidden_dim=depth_decoder_hidden_dim,
            num_layers=depth_decoder_num_layers,
        )
        self.use_priv_decoder = use_priv_decoder
        if self.use_priv_decoder:
            self.priv_decoder = StateMLP(
                in_dim=state_dim,
                out_dim=priv_decoder_out_dim,
                hidden_dim=priv_decoder_hidden_dim,
                num_layers=priv_decoder_num_layers,
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
        rt_mlps_num_layers=1,
        dropout=0.0,
        rnn_type="gru",
        encoder_name="regnet_y_800mf",
        do_predict_2d=False,
        do_predict_6d_rot=False,
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
        self.do_predict_2d = do_predict_2d
        self.do_predict_6d_rot = do_predict_6d_rot

        self.input_dim = depth_dim + rgb_dim

        if rnn_type == "lstm":
            self.lstm_cell = LSTMCell(self.input_dim, hidden_dim)
        elif rnn_type == "gru":
            self.lstm_cell = nn.GRUCell(self.input_dim, hidden_dim)
        elif rnn_type == "gru_custom":
            self.lstm_cell = GRUCell(self.input_dim, hidden_dim)
        else:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")

        self.belief_encoder = BeliefEncoder(
            self.lstm_cell,
            rnn_hidden_dim=hidden_dim,
            depth_latent_dim=depth_dim,
            belief_enc_hidden_dim=benc_belief_enc_hidden_dim,
            belief_depth_enc_hidden_dim=benc_belief_depth_enc_hidden_dim,
            belief_enc_num_layers=benc_belief_enc_num_layers,
            belief_depth_enc_num_layers=benc_belief_depth_enc_num_layers,
        )
        self.belief_decoder = BeliefDecoder(
            state_dim=hidden_dim,
            priv_decoder_out_dim=bdec_priv_decoder_out_dim,
            priv_decoder_hidden_dim=bdec_priv_decoder_hidden_dim,
            depth_decoder_out_dim=depth_dim,
            depth_decoder_hidden_dim=bdec_depth_decoder_hidden_dim,
            hidden_attn_hidden_dim=bdec_hidden_attn_hidden_dim,
        )
        if do_predict_2d:
            self.t_mlp_out_dim = 2
            self.depth_mlp = MLP(
                in_dim=depth_dim + rgb_dim,
                out_dim=1,
                hidden_dim=hidden_dim,
                num_layers=rt_mlps_num_layers,
                dropout=dropout,
            )
        else:
            self.t_mlp_out_dim = 3
        self.t_mlp = MLP(
            in_dim=depth_dim + rgb_dim,
            out_dim=self.t_mlp_out_dim,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            act_out=nn.Sigmoid() if do_predict_2d else None,  # normalized coords
            dropout=dropout,
        )
        self.rot_mlp = MLP(
            in_dim=depth_dim + rgb_dim,
            out_dim=6 if do_predict_6d_rot else 4,
            hidden_dim=hidden_dim,
            num_layers=rt_mlps_num_layers,
            dropout=dropout,
        )
        self.encoder_name = encoder_name
        self.encoder_img, self.encoder_depth = get_encoders(encoder_name)

    def forward(self, rgb, depth, hx, cx=None):

        latent_rgb = self.encoder_img(rgb)
        latent_depth = self.encoder_depth(depth)

        encoder_out = self.belief_encoder(latent_rgb, latent_depth, hx, cx)
        hx, cx = (encoder_out["hidden_state"], encoder_out["cx"])
        decoder_out = self.belief_decoder(hx, latent_depth)

        # pose estimation
        belief_state = encoder_out["belief_state"]
        extracted_obs = torch.cat([latent_rgb, belief_state], dim=1)
        t = self.t_mlp(extracted_obs)
        rot = self.rot_mlp(extracted_obs)

        res = {
            "encoder_out": encoder_out,
            "decoder_out": decoder_out,
            "latent_depth": latent_depth,
            "hx": hx,
            "cx": cx,
            "t": t,
            "rot": rot,
        }
        if self.do_predict_2d:
            center_depth = self.depth_mlp(extracted_obs)
            res["center_depth"] = center_depth

        return res
