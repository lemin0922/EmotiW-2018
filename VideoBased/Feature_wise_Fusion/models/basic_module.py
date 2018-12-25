import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from six.moves import xrange
import math


class LSTM_my(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True):
        super(LSTM_my, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        # TODO : dropout
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.input_weights.weight.uniform_(-stdv, stdv)
        self.input_weights.bias.fill_(0)
        self.hidden_weights.weight.uniform_(-stdv, stdv)
        self.hidden_weights.bias.fill_(0)

    def forward(self, x, hidden):
        # TODO : implement bidirection lstm
        def recurrence(input_, hidden_):
            prev_h, prev_c = hidden_
            gates = self.input_weights(input_) + self.hidden_weights(prev_h)
            in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

            in_gate = F.sigmoid(in_gate)
            forget_gate = F.sigmoid(forget_gate)
            cell_gate = F.tanh(cell_gate)
            out_gate = F.sigmoid(out_gate)

            next_c = (forget_gate * prev_c) + (in_gate * cell_gate)
            next_h = out_gate * F.tanh(next_c)

            # TODO : dropout

            return next_h, next_c

        if self.batch_first:
            x = x.transpose(0, 1)

        output = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden)
            output.append(hidden[0]) if isinstance(hidden, tuple) else output.append(hidden)

        output = torch.cat(output, 0).view(x.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

class RNNSoftPlus(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True):
        super(RNNSoftPlus, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_bidirectional = 2 if bidirectional else 1

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.softplus = nn.Softplus(beta=5)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.i2h.weight.data.uniform_(-stdv, stdv)
        self.h2h.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        def recurrence(input_, hidden_):
            i2h = self.i2h(input_)
            h2h = self.h2h(hidden_)
            next_h = self.softplus(i2h + h2h)
            return next_h

        if self.batch_first:
            x = x.transpose(0, 1)

        if self.bidirectional:
            bi_hidden = hidden[1]
            hidden = hidden[0]

        outputs = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden)
            outputs.append(hidden)

        if self.bidirectional:
            bi_outputs = []
            for i in reversed(xrange(x.size(0))):
                bi_hidden = recurrence(x[i], bi_hidden)
                bi_outputs.append(torch.cat([outputs[i-16], bi_hidden], 1).unsqueeze(0))

        out = torch.cat(bi_outputs, 0) if self.bidirectional else torch.cat(outputs, 0)

        if self.bidirectional:
            out = out.transpose(0, 1)

        return out

class GRU_my(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, batch_first=True, bidirectional=True):
        super(GRU_my, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # parameter
        self.i2h_zr = nn.Linear(input_size, 2 * hidden_size)
        self.h2h_zr = nn.Linear(hidden_size, 2 * hidden_size)
        self.i2h_tilda = nn.Linear(input_size, hidden_size)
        self.h2h_tilda = nn.Linear(hidden_size, hidden_size)

        # non-learnable parameter
        self.one = nn.Parameter(torch.ones(1), requires_grad=False)

    def init_weights(self):
        pass

    def forward(self, x, hidden):
        def recurrence(input_, hidden_):
            i2h_tilda = self.i2h_tilda(input_)
            h2h_tilda = self.h2h_tilda(hidden_)

            zr_gate = self.i2h_zr(input_) + self.h2h_zr(hidden_)
            r_gate, z_gate = zr_gate.chunk(2, 1)

            r_gate = F.sigmoid(r_gate)
            z_gate = F.sigmoid(z_gate)
            h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)
            next_h = z_gate * h_tilda + (self.one.expand(z_gate.size()) - z_gate) * hidden_

            return next_h

        if self.batch_first:
            x = x.transpose(0, 1)

        if self.bidirectional:
            bi_hidden = hidden[1]
            hidden = hidden[0]

        outputs = []
        for i in xrange(x.size(0)):
            hidden = recurrence(x[i], hidden)
            outputs.append(hidden)

        if self.bidirectional:
            bi_outputs = []
            for i in reversed(xrange(x.size(0))):
                bi_hidden = recurrence(x[i], bi_hidden)
                bi_outputs.append(torch.cat([outputs[i-16], bi_hidden], 1).unsqueeze(0))

        out = torch.cat(bi_outputs, 0) if self.bidirectional else torch.cat(outputs, 0)

        if self.bidirectional:
            out = out.transpose(0, 1)

        return out