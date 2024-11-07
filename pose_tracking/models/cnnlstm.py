import torch
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


if __name__ == "__main__":
    model = LSTMCell(10, 20)
    input = torch.randn(6, 10)
    hx = torch.randn(6, 20)
    cx = torch.randn(6, 20)
    output = model(input, hx, cx)
    print(output[0].shape)


class BeliefEncoder(nn.Module):
    def __init__(self, rnn_cell, rnn_hidden_dim, depth_latent_dim, belief_prior_dim, belief_posterior_dim):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.depth_latent_dim = depth_latent_dim
        self.belief_prior_dim = belief_prior_dim
        self.belief_posterior_dim = belief_posterior_dim

        self.belief_prior_mlp = StateMLP(in_dim=rnn_hidden_dim, out_dim=belief_prior_dim, hidden_dim=100, num_layers=2)
        self.belief_depth_mlp = StateMLP(
            in_dim=rnn_hidden_dim, out_dim=belief_posterior_dim, hidden_dim=100, num_layers=2
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
        in_dim,
        priv_decoder_out_dim,
        depth_decoder_hidden_dim,
        priv_decoder_hidden_dim,
        depth_decoder_out_dim,
        priv_decoder_num_layers=1,
        depth_decoder_num_layers=1,
    ):
        super().__init__()
        self.priv_decoder = StateMLP(
            in_dim,
            priv_decoder_out_dim,
            priv_decoder_hidden_dim,
            num_layers=priv_decoder_num_layers,
        )
        self.hidden_encoder = nn.Sequential(nn.Linear(hidden_dim, depth_decoder_out_dim), nn.Sigmoid())
        self.depth_decoder = StateMLP(
            in_dim,
            depth_decoder_out_dim,
            depth_decoder_hidden_dim,
            num_layers=depth_decoder_num_layers,
        )

    def forward(self, ht, depth_latent):
        attn = self.hidden_encoder(ht)
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


