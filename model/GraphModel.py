import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class GNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = 2 * self.hidden_size
        self.gate_size = 3 * self.hidden_size

        # some params
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):  ###hidden[16,3,200]
        # A (batch, node_num, 2*node_num)
        # hidden (batch, node_num, embd-size)
        print(A.shape)

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah ##[16,3,200]
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)  # (batch, node_num, 2*hidden_size) [16,2,400]

        gi = F.linear(inputs, self.w_ih, self.b_ih)  # (batch, node_num, 3*hidden_size)
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # (batch, node_num, 3*hidden_size)

        i_r, i_i, i_n = gi.chunk(3, 2)  #  (batch, node_num, hidden_size)
        h_r, h_i, h_n = gh.chunk(3, 2)  #  (batch, node_num, hidden_size)
        resetgate = torch.sigmoid(i_r + h_r)  # (batch, node_num, hidden_size)
        inputgate = torch.sigmoid(i_i + h_i)  # (batch, node_num, hidden_size)

        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate) # (batch, node_num, hidden_size)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden




class SessionGraph(nn.Module):
    def __init__(self, config, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = config['hiddenSize']
        self.n_node = n_node
        self.nonhybrid = config['nonhybrid']
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=config['step'])
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.reset_parameters()
        self.linear_five = nn.Linear(self.hidden_size*2, 1, bias=False)
        self.linear_six = nn.Linear(768, self.hidden_size, bias=True)


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, A):
        # hidden = self.linear_six(inputs)  ##output:[16,3,200]
        # hidden = self.gnn(A, hidden)
        hidden = self.embedding(inputs)
        # int = hidden
        hidden = self.gnn(A, hidden)
        # hidden = int + hidden
        return hidden

    def compute_globalhidden(self, hidden, mask,num):
        # ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        # q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1]) # (batch_size, 1, latent_size)
        q2 = self.linear_two(hidden) # (batch_size, seq_length, latent_size)

        # old part
        # alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        # new part
        # adding attention
        alpha = self.linear_three(torch.sigmoid(q2)) #[batch,node_num,1]
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        q1 = self.linear_one(a).view(a.shape[0], 1, a.shape[1])
        beta = self.linear_four(torch.sigmoid(q1 + q2))
        newa = torch.sum(beta * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  ##[batch,hidden]

        if not self.nonhybrid:
            alpha = torch.sigmoid(self.linear_five(torch.cat([a, newa], 1))).view(num,1)
            a = alpha*newa + (1-alpha)*a
            # a = self.linear_transform(torch.cat([a, newa], 1))
        return a

