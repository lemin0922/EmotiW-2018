import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DecoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ctx_hidden_size,
                 fc_hidden_size,
                 num_classes=7,
                 mode='lstm',
                 add_fc=True,
                 ):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_hidden_size = ctx_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.add_fc = add_fc
        self.mode = mode

        if mode == 'lstm':
            self.lstm = ContextLSTM(input_size, hidden_size, ctx_hidden_size, batch_first=True)
        elif mode == 'gru':
            self.lstm = ContextGRU(input_size, hidden_size, ctx_hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        if add_fc:
            self.fc = nn.Linear(fc_hidden_size, num_classes)
            self.fc_add = nn.Linear(hidden_size, fc_hidden_size)
            self.relu = nn.ReLU()
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def get_init_states(self, input_, mode):
        batch_size = input_.size(0)
        h_init = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        if mode == 'lstm':
            c_init = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
            return h_init, c_init
        else:
            return h_init

    def forward(self, v_features, i_features):
        init_states = self.get_init_states(v_features, self.mode)

        x, _ = self.lstm(i_features, init_states, v_features)
        x = self.dropout(x)
        if self.add_fc:
            x = self.relu(self.fc_add(x[:, -1, :]))
            x = self.dropout(x)
            out = self.fc(x)
        else:
            out = self.fc(x[:, -1, :])
        return out

class ContextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_hidden_size, batch_first=True):
        super(ContextLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        # to split hidden_size for 4 gates , so multiply hidden_size by 4
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        self.context_weights = nn.Linear(ctx_hidden_size, 4 * hidden_size)

    def forward(self, input, hidden, ctx):
        def recurrence(input, hidden, ctx):
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                    self.hidden_weights(hx) + \
                    self.context_weights(ctx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

class ContextGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ctx_hidden_size, batch_first=True):
        super(ContextGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        # to split hidden_size for 4 gates , so multiply hidden_size by 4
        self.i2h_zr = nn.Linear(input_size, 2 * hidden_size)
        self.h2h_zr = nn.Linear(hidden_size, 2 * hidden_size)
        self.i2h_tilda = nn.Linear(input_size, hidden_size)
        self.h2h_tilda = nn.Linear(hidden_size, hidden_size)
        self.ctx2h = nn.Linear(ctx_hidden_size, 2 * hidden_size)

    def forward(self, input, hidden, ctx):
        def recurrence(input, hidden, ctx):
            hx = hidden  # n_b x hidden_dim
            i2h_tilda = self.i2h_tilda(input)
            h2h_tilda = self.h2h_tilda(hidden)
            zr_gate = self.i2h_zr(input) + self.h2h_zr(hidden) + self.ctx2h(ctx)
            r_gate, z_gate = zr_gate.chunk(2, 1)

            r_gate = F.sigmoid(r_gate)
            z_gate = F.sigmoid(z_gate)
            h_tilda = F.tanh(i2h_tilda + r_gate * h2h_tilda)

            hy = z_gate * h_tilda + (1 - z_gate) * hx
            return hy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden, ctx)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden
