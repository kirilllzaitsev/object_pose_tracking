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
if __name__ == "__main__":
    model = LSTMCell(10, 20)
    input = torch.randn(6, 10)
    hx = torch.randn(6, 20)
    cx = torch.randn(6, 20)
    output = model(input, hx, cx)
    print(output[0].shape)
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
        hidden_dim,
        priv_decoder_out_dim=20,
        priv_decoder_hidden_dim=10,
        depth_decoder_hidden_dim=10,
        depth_decoder_out_dim=50,
    )
    outputs = belief_decoder(inputs, depth_latent)
    for k, v in outputs.items():
        print(k, v.shape)
