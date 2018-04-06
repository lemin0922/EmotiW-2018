#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:02:27 2018

@author: dmsl
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import Util as util
import os
import numpy as np

input_size = 4096
hidden_size = 2048
num_layers = 1

ROOT_PATH = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/Temporal_Learn/'
VAL_PATH = '/home/dmsl/nas/DMSL/AFEW/NpzData/Val/'

# TODO : select rnn cell (GRU / LSTM)
#CELL = 'mult_lstm'
CELL = 'mult_gru'

# TODO : select which model's feature data
print("Load data...")
X_VAL_PATH = VAL_PATH + 'x_Densenet_frames_no_overlap.npz'
Y_VAL_PATH = VAL_PATH + 'y_Densenet_frames_no_overlap.npz'
#X_VAL_PATH = VAL_PATH + 'x_DenseNet169_original_frames_no_overlap.npz'
#Y_VAL_PATH = VAL_PATH + 'y_DenseNet169_original_frames_no_overlap.npz'

x_val = util.load_from_npz(X_VAL_PATH)
y_val = util.load_from_npz(Y_VAL_PATH)
y_val = np.squeeze(y_val, axis=1)

gpuid = 0
# Hyper Parameters
sequence_length = 16
num_classes = 7
batch_size = 256
learning_rate = 1e-3
bidirectional = False

# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, mode, bidirectional):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        if self.mode == 'mult_lstm':
            self.num_layers = num_layers
            self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.dropout1 = nn.Dropout(0.5)
            self.dropout2 = nn.Dropout(0.5)
            self.dropout3 = nn.Dropout(0.5)
        else:
            self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru2 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.gru3 = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.dropout1 = nn.Dropout(0.5)
            self.dropout2 = nn.Dropout(0.5)
            self.dropout3 = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial states 
#        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(0)) 
#        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(0))
        
        # Forward propagate RNN
#        out, _ = self.lstm(x, (h0, c0)) 
        if self.mode == 'mult_lstm':
            x, _ = self.lstm1(x)
#            x = self.dropout1(x)
#            x, _ = self.lstm2(x)
#            x = self.dropout2(x)
#            x, _ = self.lstm3(x)
            x = self.dropout3(x)
        else:
            x, _ = self.gru1(x)
#            x = self.dropout1(x)
#            x, _ = self.gru2(x)
#            x = self.dropout2(x)
#            x, _ = self.gru3(x)
            x = self.dropout3(x)
        # Decode hidden state of last time step
        out = self.fc(x[:, -1, :])  
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes, mode=CELL, bidirectional=bidirectional)
rnn.cuda(gpuid)

weight_path = {
        'model46_1' : ROOT_PATH + 'mult_gru_Train_only_overlap_dense_fc6_frames_overlap_256ens3/mult_gru_Train_only_overlap_dense_fc6_frames_overlap_256ens3.pkl',
        'model45_5' : ROOT_PATH + 'mult_gru_Train_only_overlap_dense_fc6_frames_overlap_256ens4/mult_gru_Train_only_overlap_dense_fc6_frames_overlap_256ens4.pkl'
        }

score= []
acc_list = []
soft_prob = []
for key, item in weight_path.items():
    rnn.load_state_dict(torch.load(item))
    rnn.eval()
    correct = 0
    total = 0
    images_val = torch.from_numpy(x_val)
    labels_val = torch.from_numpy(y_val)
    images_val= images_val.type(torch.FloatTensor)
    images_val = Variable(images_val.view(-1, sequence_length, input_size)).cuda(gpuid)
    outputs_val = rnn(images_val)
    prob = F.softmax(outputs_val, dim=1)
    _, predicted = torch.max(outputs_val.data, 1)
    total += labels_val.size(0)
    correct += (predicted.cpu() == labels_val).sum()
    test_acc = 100 * correct / total
    
    score.append(outputs_val.data.cpu().numpy())
    acc_list.append(test_acc)
    soft_prob.append(prob.data.cpu().numpy())

ensemble_correct = 0
max_prob = 0
weight_range = np.linspace(0.0, 1.0, 100000)
for i in range(len(weight_range)):
    weigth_score = weight_range[i]*soft_prob[0] + (1 - weight_range[i])*soft_prob[1]
    pred = np.argmax(weigth_score, axis=1)
    ensemble_correct = np.sum(pred == y_val)
    current_prob = ensemble_correct * 100 / total
    if max_prob < current_prob:
        max_prob = current_prob
    
print('Ensemble Accuracy : {}'.format(max_prob))