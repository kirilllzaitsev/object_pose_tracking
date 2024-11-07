import torch
import torch.nn as nn
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
        belief_posterior_dim,
        belief_enc_hidden_dim,
        belief_depth_enc_hidden_dim,
        belief_enc_num_layers=2,
        belief_depth_enc_num_layers=2,
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.depth_latent_dim = depth_latent_dim
        self.belief_posterior_dim = belief_posterior_dim

        self.belief_prior_mlp = StateMLP(
            in_dim=rnn_hidden_dim,
            out_dim=belief_posterior_dim,
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
    ):
        super().__init__()
        self.priv_decoder = StateMLP(
            in_dim=state_dim,
            out_dim=priv_decoder_out_dim,
            hidden_dim=priv_decoder_hidden_dim,
            num_layers=priv_decoder_num_layers,
        )
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

    def forward(self, ht, depth_latent):
        attn = self.hidden_attn(ht)
        depth_latent_attn = depth_latent * attn
        depth_decoded = self.depth_decoder(ht)
        depth_final = depth_latent_attn + depth_decoded
        priv_decoded = self.priv_decoder(ht)
        return {
            "attn": attn,
            "depth_latent_attn": depth_latent_attn,
            "depth_decoded": depth_decoded,
            "depth_final": depth_final,
            "priv_decoded": priv_decoded,
        }


class RecurrentCNN(nn.Module):
    def __init__(
        self,
        depth_dim,
        rgb_dim,
        hidden_dim,
        benc_depth_latent_dim,
        benc_belief_posterior_dim,
        benc_belief_enc_hidden_dim,
        benc_belief_depth_enc_hidden_dim,
        bdec_priv_decoder_out_dim,
        bdec_priv_decoder_hidden_dim,
        bdec_depth_decoder_out_dim,
        bdec_depth_decoder_hidden_dim,
        bdec_hidden_attn_hidden_dim,
        benc_belief_enc_num_layers=2,
        benc_belief_depth_enc_num_layers=2,
        rnn_type="gru",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.input_dim = rgb_dim + depth_dim
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim

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
            depth_latent_dim=benc_depth_latent_dim,
            belief_posterior_dim=benc_belief_posterior_dim,
            belief_enc_hidden_dim=benc_belief_enc_hidden_dim,
            belief_depth_enc_hidden_dim=benc_belief_depth_enc_hidden_dim,
            belief_enc_num_layers=benc_belief_enc_num_layers,
            belief_depth_enc_num_layers=benc_belief_depth_enc_num_layers,
        )
        self.belief_decoder = BeliefDecoder(
            state_dim=hidden_dim,
            priv_decoder_out_dim=bdec_priv_decoder_out_dim,
            priv_decoder_hidden_dim=bdec_priv_decoder_hidden_dim,
            depth_decoder_out_dim=bdec_depth_decoder_out_dim,
            depth_decoder_hidden_dim=bdec_depth_decoder_hidden_dim,
            hidden_attn_hidden_dim=bdec_hidden_attn_hidden_dim,
        )

    def forward(self, inputs, hx_0, cx_0=None):
        # inputs: seq of dicts with keys "rgb" and "depth"
        decoder_outputs = []
        encoder_outputs = []
        hx, cx = hx_0, cx_0

        for input_t in inputs:
            rgb_latent, depth_latent = input_t["rgb"], input_t["depth"]
            encoder_out = self.belief_encoder(rgb_latent, depth_latent, hx=hx, cx=cx)
            hx, cx = (encoder_out["hidden_state"], encoder_out["cx"])
            decoder_out = self.belief_decoder(encoder_out["hidden_state"], depth_latent)
            decoder_outputs.append(decoder_out)
            encoder_outputs.append(encoder_out)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
        }


if __name__ == "__main__":
    model = LSTMCell(10, 20)
    input = torch.randn(6, 10)
    hx = torch.randn(6, 20)
    cx = torch.randn(6, 20)
    output = model(input, hx, cx)
    print(output[0].shape)
    batch_size = 6
    hidden_dim = 20
    inputs = torch.randn(batch_size, hidden_dim)
    depth_latent = torch.randn(batch_size, 50)
    belief_decoder = BeliefDecoder(
        state_dim=hidden_dim,
        priv_decoder_out_dim=20,
        priv_decoder_hidden_dim=10,
        depth_decoder_hidden_dim=10,
        depth_decoder_out_dim=50,
        hidden_attn_hidden_dim=20,
    )
    outputs = belief_decoder(inputs, depth_latent)
    for k, v in outputs.items():
        print(k, v.shape)

    hidden_dim = 20
    batch_size = 3
    rgb_dim = 10
    depth_dim = 50
    priv_dim = 20
    net = RecurrentCNN(
        depth_dim=depth_dim,
        rgb_dim=rgb_dim,
        hidden_dim=hidden_dim,
        rnn_type="gru",
        benc_depth_latent_dim=50,
        benc_belief_posterior_dim=50,
        bdec_priv_decoder_out_dim=priv_dim,
        bdec_priv_decoder_hidden_dim=40,
        bdec_depth_decoder_out_dim=50,
        bdec_depth_decoder_hidden_dim=80,
        benc_belief_enc_hidden_dim=110,
        benc_belief_depth_enc_hidden_dim=200,
        bdec_hidden_attn_hidden_dim=128,
    )
    seq_len = 6
    inputs = [
        {"rgb": torch.randn(batch_size, rgb_dim), "depth": torch.randn(batch_size, depth_dim)} for _ in range(seq_len)
    ]
    hx = torch.zeros(batch_size, hidden_dim)
    cx = torch.zeros(batch_size, hidden_dim)
    outputs = net(inputs, hx)
    for k, v in outputs.items():
        print(k, len(v), v[0].keys())
        for out_t in v:
            for k2, v2 in out_t.items():
                if v2 is None:
                    continue
                print(k2, v2.shape)
            break
