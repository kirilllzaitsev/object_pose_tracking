import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_dim))

        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_dim) if self.hidden_dim > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class LSTMCell(RNNCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, input, state):
        hx, cx = state

        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cx = (forgetgate * cx) + (ingate * cellgate)
        hx = outgate * torch.tanh(cx)

        return {"hidden_state": hx, "cell_state": cx}


class GRUCell(RNNCell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, input, state):
        hx = state

        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh

        resetgate, updategate, newgate = gates.chunk(3, 1)

        resetgate = torch.sigmoid(resetgate)
        updategate = torch.sigmoid(updategate)
        newgate = torch.tanh(newgate)

        hx = (1 - updategate) * hx + updategate * newgate

        return {"hidden_state": hx}
