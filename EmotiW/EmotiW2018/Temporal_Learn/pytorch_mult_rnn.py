#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:32:50 2018

@author: dmsl
"""

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util import Util as util
import os
import numpy as np

class DataSet(object):

  def __init__(self,
               images,
               labels):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]
  

# TODO : check parameter
training_steps = 30000

# TODO : check data
train_type = 'Train_only'
#train_type = 'Train_all'
#train_type = 'Train_all2'
bidirectional = False
overlap = True

# TODO : check hyperparameter
input_size = 4096
hidden_size = 2048
num_layers = 1

ROOT_PATH = '/home/dmsl/Desktop/EmotiW/EmotiW2018/ImageBased/Temporal_Learn/'
TRAIN_PATH = '/home/dmsl/nas/DMSL/AFEW/NpzData/' + train_type + '/'
VAL_PATH = '/home/dmsl/nas/DMSL/AFEW/NpzData/Val/'

# TODO : select rnn cell (GRU / LSTM)
CELL = 'mult_lstm'
#CELL = 'mult_gru'

# TODO : check version
name = '_single_2048ens1'

# TODO : gpu control
gpuid = 0

# TODO : select which model's feature data
print("Load data...")
if overlap:
#    data_name = '_DenseNet169_original_frames_overlap'
#    data_name = '_dense_fc6_frames_overlap'
    data_name = '_DenseNet169_fc2_fc1_overlap'
    X_TRAIN_PATH = TRAIN_PATH + 'x' + data_name + '.npz'
    Y_TRAIN_PATH = TRAIN_PATH + 'y' + data_name + '.npz'
else:
#    data_name = '_DenseNet169_original_frames_no_overlap'
#    data_name = '_dense_fc6_frames_nooverlap'
    data_name = '_DenseNet169_fc2_fc1_no_overlap'
    X_TRAIN_PATH = TRAIN_PATH + 'x' + data_name + '.npz'
    Y_TRAIN_PATH = TRAIN_PATH + 'y' + data_name + '.npz'

X_VAL_PATH = VAL_PATH + 'x_DenseNet169_fc2_fc1_no_overlap.npz'
Y_VAL_PATH = VAL_PATH + 'y_DenseNet169_fc2_fc1_no_overlap.npz'
#X_VAL_PATH = VAL_PATH + 'x_DenseNet169_original_frames_no_overlap.npz'
#Y_VAL_PATH = VAL_PATH + 'y_DenseNet169_original_frames_no_overlap.npz'

x_train, x_val = util.load_from_npz(X_TRAIN_PATH), util.load_from_npz(X_VAL_PATH)
y_train, y_val = util.load_from_npz(Y_TRAIN_PATH), util.load_from_npz(Y_VAL_PATH)
y_train, y_val = np.squeeze(y_train, axis=1), np.squeeze(y_val, axis=1)
x_frame_data = DataSet(x_train, y_train)

# TODO : check save path
if overlap:
    if bidirectional:
        DIR_NAME = CELL + '_' + train_type + '_bi_overlap' + data_name  + name
    else:
        DIR_NAME = CELL + '_' + train_type + '_overlap' + data_name +  name
else:
    if bidirectional:
        DIR_NAME = CELL + '_' + train_type + '_bi_no_overlap' + data_name + name
    else:
        DIR_NAME = CELL + '_' + train_type + '_no_overlap' + data_name + name

SAVE_PATH = ROOT_PATH + DIR_NAME
MODEL_PATH = SAVE_PATH + '/' + DIR_NAME + '.pkl'
print("Save path : " + MODEL_PATH)

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# Hyper Parameters
sequence_length = 16
num_classes = 7
batch_size = 256
learning_rate = 1e-3

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
            x = self.dropout1(x)
            x, _ = self.gru2(x)
            x = self.dropout2(x)
            x, _ = self.gru3(x)
            x = self.dropout3(x)
        # Decode hidden state of last time step
        out = self.fc(x[:, -1, :])  
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes, mode=CELL, bidirectional=bidirectional)
rnn.cuda(gpuid)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=600, verbose=True, min_lr=1e-7)
# Train the Model
print("Train Start")
max_acc = 0
for step in range(training_steps):
    rnn.train()
    images, labels = x_frame_data.next_batch(batch_size)
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images = images.type(torch.FloatTensor)
    images = Variable(images.view(-1, sequence_length, input_size)).cuda(gpuid)  
    labels = Variable(labels).cuda(gpuid)
    
    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = rnn(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (step+1) % 500 == 0:
        print ('Step [%d/%d], Loss: %.4f' 
               %(step+1, training_steps, loss.data[0]))
    if step % (x_train.shape[0] // batch_size) == 0:
        rnn.eval()
        correct = 0
        total = 0
        images_val = torch.from_numpy(x_val)
        labels_val = torch.from_numpy(y_val)
        images_val= images_val.type(torch.FloatTensor)
        images_val = Variable(images_val.view(-1, sequence_length, input_size)).cuda(gpuid)
        outputs_val = rnn(images_val)
        _, predicted = torch.max(outputs_val.data, 1)
        total += labels_val.size(0)
        correct += (predicted.cpu() == labels_val).sum()
        test_acc = 100 * correct / total
        
        if max_acc < test_acc:
            print("Save " + DIR_NAME + " - performance improved from {:.6f} to {:.6f}".format(max_acc, test_acc))
            max_acc = test_acc
            torch.save(rnn.state_dict(), MODEL_PATH)
        scheduler.step(max_acc)
# Save the Model

# save configuration
save_dict = {
    'Test_Accuracy' : str(max_acc),
    'Cell_type' : CELL,
    'Batch_size' : str(batch_size),
    'Hidden_unit' : str(hidden_size),
    'X_train' : X_TRAIN_PATH.split('/')[-1],
    'Y_train' : Y_TRAIN_PATH.split('/')[-1]
}
with open(SAVE_PATH + '/config.txt', 'w') as f:
    for key, content in sorted(save_dict.items()):
        f.write(key + ' : ' + content + '\n')