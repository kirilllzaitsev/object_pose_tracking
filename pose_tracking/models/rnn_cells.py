import numpy as np
import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Define the LSTM cell parameters
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))

        self.init_parameters(hidden_dim)

    def init_parameters(self, hidden_dim):
        nn.init.kaiming_uniform_(self.weight_ih, a=np.sqrt(hidden_dim))
        nn.init.kaiming_uniform_(self.weight_hh, a=np.sqrt(hidden_dim))

    def forward(self, input, state):
        hx, cx = state  # Unpack previous hidden and cell states

        # Linear transformations for the gates
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh

        # Split gates into 4 parts: input gate, forget gate, cell gate, output gate
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # Apply activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # Update cell state and hidden state
        cx = (forgetgate * cx) + (ingate * cellgate)  # New cell state
        hx = outgate * torch.tanh(cx)  # New hidden state

        return {"hidden_state": hx, "cell_state": cx}


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Define the GRU cell parameters
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_dim))

        self.init_parameters(hidden_dim)

    def init_parameters(self, hidden_dim):
        nn.init.kaiming_uniform_(self.weight_ih, a=np.sqrt(hidden_dim))
        nn.init.kaiming_uniform_(self.weight_hh, a=np.sqrt(hidden_dim))

    def forward(self, input, state):
        hx = state  # Unpack previous hidden state

        # Linear transformations for the gates
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh

        # Split gates into 3 parts: reset gate, update gate, new gate
        resetgate, updategate, newgate = gates.chunk(3, 1)

        # Apply activations
        resetgate = torch.sigmoid(resetgate)
        updategate = torch.sigmoid(updategate)
        newgate = torch.tanh(newgate)

        # Update hidden state
        hx = (1 - updategate) * hx + updategate * newgate

        return {"hidden_state": hx}
