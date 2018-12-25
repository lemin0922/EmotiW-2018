import torch
import torch.nn as nn
from torch.autograd import Variable

from models.C3D import C3D
# from EmotiW2018.VideoBased.Feature_wise_Fusion.C3D import C3D

WEIGHT_PATH = '/home/dmsl/nas/HDD_server6/EmotiW/data/C3D/C3D_keras_to_pytorch.pkl'

class EncoderC3D(nn.Module):
    def __init__(self, embed_size):
        super(EncoderC3D, self).__init__()
        base_model = C3D().cuda()
        base_model.load_state_dict(torch.load(WEIGHT_PATH))

        self.conv1 = base_model.conv1
        self.pool1 = base_model.pool1

        self.conv2 = base_model.conv2
        self.pool2 = base_model.pool2

        self.conv3a = base_model.conv3a
        self.conv3b = base_model.conv3b
        self.pool3 = base_model.pool3

        self.conv4a = base_model.conv4a
        self.conv4b = base_model.conv4b
        self.pool4 = base_model.pool4

        self.conv5a = base_model.conv5a
        self.conv5b = base_model.conv5b
        self.pool5 = base_model.pool5

        self.fc6 = base_model.fc6
        self.fc7 = base_model.fc7
        self.fc8 = nn.Linear(4096, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, x):
        # TODO : freeze c3d network using torch.no_grad()
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h = self.fc8(h)
        feature = self.bn(h)

        return feature

class DecoderRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 fc_hidden_size,
                 num_layers,
                 num_classes=7,
                 cell='gru',
                 add_fc=True,
                 bidirectional=False
                 ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_layers = num_layers
        self.cell = cell
        self.add_fc = add_fc
        self.bidirectional = bidirectional
        self.num_bidirectional = 2 if bidirectional else 1

        if self.cell == 'lstm':
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif self.cell == 'gru':
            self.gru = nn.GRU(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        self.dropout = nn.Dropout(p=0.5)

        if add_fc:
            self.fc = nn.Linear(fc_hidden_size, num_classes)
            self.fc_add = nn.Linear(hidden_size*self.num_bidirectional, fc_hidden_size)
            self.relu = nn.ReLU()
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, v_features, i_features):
        h0 = v_features.unsqueeze(0)
        if self.cell == 'lstm':
            c0 = Variable(
                torch.zeros(
                    self.num_layers*self.num_bidirectional,
                    v_features.size(0),
                    self.hidden_size)
            ).cuda()
            # TODO : Using both nn.DataParallel & flatten_parameters() dont work
            # TODO: issue#7092(https://github.com/pytorch/pytorch/issues/7092)
            # self.lstm.flatten_parameters()
            x, _ = self.lstm(i_features, (h0, c0))
        elif self.cell == 'gru':
            # self.gru.flatten_parameters()
            x, _ = self.gru(i_features, h0)

        x = self.dropout(x)
        if self.add_fc:
            x = self.relu(self.fc_add(x[:, -1, :]))
            x = self.dropout(x)
            out = self.fc(x)
        else:
            out = self.fc(x[:, -1, :])
        return out

class DecoderConcatRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ctx_hidden_size,
                 fc_hidden_size,
                 num_layers,
                 num_classes=7,
                 cell='gru',
                 add_fc=True,
                 bidirectional=False
                 ):
        super(DecoderConcatRNN, self).__init__()
        self.hidden_size = hidden_size
        self.ctx_hidden_size = ctx_hidden_size
        self.fc_hidden_size = fc_hidden_size
        self.num_layers = num_layers
        self.cell = cell
        self.add_fc = add_fc
        self.bidirectional = bidirectional
        self.num_bidirectional = 2 if bidirectional else 1

        if self.cell == 'lstm':
            self.lstm = nn.LSTM(
                input_size + ctx_hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        elif self.cell == 'gru':
            self.gru = nn.GRU(
                input_size + ctx_hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional
            )
        self.dropout = nn.Dropout(p=0.5)

        if add_fc:
            self.fc = nn.Linear(fc_hidden_size, num_classes)
            self.fc_add = nn.Linear(hidden_size*self.num_bidirectional, fc_hidden_size)
            self.relu = nn.ReLU()
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, v_features, i_features):
        h0 = Variable(
           torch.zeros(
               self.num_layers*self.num_bidirectional,
               v_features.size(0),
              self.hidden_size)
        ).cuda()

        # feat_concat = torch.cat((v_features.unsqueeze(1), i_features), 1) # method 1 : concatenate per frame feature, shape : (B, 17, 4096)
        feat_concat = torch.cat((v_features.unsqueeze(1).repeat(1, 16, 1), i_features), 2)  # method 3 : concatenate all sequence, shape : (B, 16, 4096 + a)
        if self.cell == 'lstm':
            c0 = Variable(
                torch.zeros(
                    self.num_layers*self.num_bidirectional,
                    v_features.size(0),
                    self.hidden_size)
            ).cuda()
            # TODO : Using both nn.DataParallel & flatten_parameters() dont work
            # TODO: issue#7092(https://github.com/pytorch/pytorch/issues/7092)
            # self.lstm.flatten_parameters()
            x, _ = self.lstm(feat_concat, (h0, c0))
        elif self.cell == 'gru':
            # self.gru.flatten_parameters()
            x, _ = self.gru(feat_concat, h0)

        x = self.dropout(x)
        if self.add_fc:
            x = self.relu(self.fc_add(x[:, -1, :]))
            x = self.dropout(x)
            out = self.fc(x)
        else:
            out = self.fc(x[:, -1, :])
        return out
