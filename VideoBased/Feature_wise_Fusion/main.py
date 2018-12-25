import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import get_loader
from models.simple_model import DecoderRNN, DecoderConcatRNN
from models.model_ctx_lstm import DecoderLSTM
from models.model_3d_cnn import EncoderC3D, EncoderResnext3D, EncoderResnet3D

import os
import copy
import time
import numpy as np
import argparse
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("-c", "--connect_type", type=str, default='B', help='Connection type : A(Add term), B(h0=3D CNN feature), C(Concatenation)')
args.add_argument("-e", "--encoder", type=str, default='c3d', help='Encoder model : c3d, resnet, resnext')
args.add_argument("-d", "--decoder", type=str, default='gru', help='Decoder model : lstm, gru')
args.add_argument("-b", "--batch_size", type=int, default=32, help='Batch size')
args.add_argument("-hs", "--h_size", type=int, default=2048, help='Hidden size')
args.add_argument("-cs", "--c_size", type=int, default=2048, help='Context hidden size')
args.add_argument("-fs", "--f_size", type=int, default=1024, help='Fully connected hidden size')
args.add_argument("-a", "--aux", type=float, default=0.3, help='auxilirary lambda value')
args.add_argument("-bn", "--batch_norm", type=int, default=0, help="use batch normalization")
args.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="use batch normalization")

args.add_argument("-n_e", "--epochs", type=int, default=1000, help="# of epochs")
args.add_argument("-n_cls", "--num_cls", type=int, default=7, help="# of classes")
args.add_argument("-l", "--reg_lambda", type=float, default=1e-4, help="lambda for regularization")
args.add_argument("-n_f", "--num_frames", type=int, default=16, help="length of video")
args.add_argument("-n_lay", "--num_layers", type=int, default=1, help="# of rnn's layers")
args.add_argument("-i", "--input_size", type=int, default=4096, help="size of input ")

config = args.parse_args()

# HyperParameter
num_epochs = config.epochs
num_classes = config.num_cls
batch_size = config.batch_size
learning_rate = config.learning_rate
reg_lambda = config.reg_lambda
sequence_length = config.num_frames
num_layers = config.num_layers
input_size = config.input_size
hidden_size = config.h_size
ctx_hidden_size = config.c_size
fc_hidden_size = config.f_size
auxiliary_lambda = config.aux

add_fc = True
bidirectional = False
batch_norm = True if config.batch_norm == 1 else False

# Mode
model_ = config.encoder
cell = config.decoder

# GPU device
device = 0

connect_type = config.connect_type

# Select the 3D CNN
if model_ == 'c3d':
    encoder = EncoderC3D(ctx_hidden_size, batch_norm) # C3D
elif model_ == 'resnet':
    encoder = EncoderResnet3D(ctx_hidden_size, batch_norm) # Resnet 3D
elif model_ == 'resnext':
    encoder = EncoderResnext3D(ctx_hidden_size, batch_norm)  # Resnext 3D

# decoder = DecoderLSTM( # ctx_lstm model
#     input_size,
#     hidden_size,
#     ctx_hidden_size,
#     fc_hidden_size,
#     num_classes,
#     add_fc=add_fc,
#     bidirectional=bidirectional
# )

# using model_ctx_lstm
if connect_type == 'A':
    decoder = DecoderLSTM(
        input_size,
        hidden_size,
        ctx_hidden_size,
        fc_hidden_size,
        num_classes,
        mode=cell,
        add_fc=add_fc,
    )
# using simple model
elif connect_type == 'B':
    decoder = DecoderRNN(
        input_size,
        hidden_size,
        fc_hidden_size,
        num_layers,
        num_classes,
        cell=cell,
        add_fc=add_fc,
        bidirectional=bidirectional
    )
elif connect_type == 'C':
    decoder = DecoderConcatRNN(
        input_size,
        hidden_size,
        ctx_hidden_size,
        fc_hidden_size,
        num_classes,
        cell=cell,
        add_fc=add_fc
    )

encoder = nn.DataParallel(encoder)
decoder = nn.DataParallel(decoder)
encoder.to(device)
decoder.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

if model_ == 'c3d':
    if batch_norm:
        params = list(encoder.module.bn.parameters()) + list(encoder.module.fc8.parameters()) + list(decoder.parameters())  # in case of AFEW, f8
    else:
        params = list(encoder.module.fc8.parameters()) + list(decoder.parameters())  # in case of AFEW, f8

else:
    if batch_norm:
        params = list(encoder.module.bn.parameters()) + list(encoder.module.fc6.parameters()) + \
                 list(encoder.module.fc7.parameters()) + list(encoder.module.fc8.parameters()) + list(decoder.module.parameters())
    else:
        params = list(encoder.module.fc6.parameters()) + list(encoder.module.fc7.parameters()) + \
                 list(encoder.module.fc8.parameters()) + list(decoder.module.parameters())

optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.99999, patience=5, verbose=True, min_lr=1e-7)

print("Load data...")
dataloaders, dataset_sizes, y_test_labels = get_loader(batch_size=batch_size, num_workers=4)
print()


# Train the Model
def train_model(encoder, decoder, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()

    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_decoder_wts= copy.deepcopy(decoder.state_dict())
    best_acc = 0.0

    def l2_regularization(model, reg):
        for W in model.parameters():
            if reg is None:
                reg = W.norm(2)
            else:
                reg = reg + W.norm(2)
        return reg

    for epoch in range(num_epochs):
        epoch_since = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                encoder.train()  # Set model to training mode
                decoder.train()  # Set model to training mode
            else:
                encoder.eval()   # Set model to evaluate mode
                decoder.eval()  # Set model to evaluate mode
                scheduler.step(best_acc)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            collect_preds = []
            every_preds = []
            for v_inputs, i_inputs, i_labels in tqdm(dataloaders[phase]):
                v_inputs = Variable(v_inputs).cuda()
                i_inputs, i_labels = Variable(i_inputs).cuda(), Variable(i_labels).cuda()

                # zero the parameter gradients
                encoder.zero_grad()
                decoder.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    v_features, out_aux = encoder(v_inputs)
                    outputs = decoder(v_features, i_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, i_labels)
                    loss_aux = criterion(out_aux, i_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # regularization
                        # TODO : reg initialize out of the current loop
                        en_reg = None
                        de_reg = None
                        en_reg = l2_regularization(encoder, en_reg)
                        de_reg = l2_regularization(decoder, de_reg)

                        loss = loss + auxiliary_lambda * loss_aux + reg_lambda * (en_reg + de_reg) # add regularization term
                        # loss = loss + reg_lambda * (en_reg + de_reg)  # add regularization term
                        loss.backward()
                        # nn.utils.clip_grad_value_(decoder.parameters(), 10)

                        optimizer.step()

                # statistics
                running_loss += loss.item() * v_inputs.size(0)
                running_corrects += torch.sum(preds == i_labels.data)

                if phase == 'test':
                    collect_preds.append(outputs.cpu().data.numpy())
                    every_preds.append(outputs.cpu().data.numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc: # phase = 'val' or 'test'
                print("Improve {:.5f} ==> {:.5f}".format(best_acc, epoch_acc))
                best_acc = epoch_acc
                best_encoder_wts = copy.deepcopy(encoder.state_dict())
                best_decoder_wts = copy.deepcopy(decoder.state_dict())
                update_flag = True

        epoch_time_elapsed = time.time() - epoch_since
        print('Elapsed Time per epoch : {:.0f}m {:.0f}s'.format(
            epoch_time_elapsed // 60, epoch_time_elapsed % 60))

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    encoder.load_state_dict(best_encoder_wts)
    decoder.load_state_dict(best_decoder_wts)

    return encoder, decoder

def eval_model(encoder_, decoder_):
    encoder_.eval()
    decoder_.eval()

    collect = []
    with torch.no_grad():
        for v_inputs, i_inputs, i_labels in tqdm(dataloaders['test']):
            v_inputs = Variable(v_inputs).cuda()
            i_inputs, i_labels = Variable(i_inputs).cuda(), Variable(i_labels).cuda()

            v_features, _ = encoder(v_inputs)
            outputs = decoder(v_features, i_inputs)
            _, preds = torch.max(outputs, 1)

            collect.append(outputs.cpu().data.numpy())

if __name__ == '__main__':
    print("===== Train Start =====")
    best_encoder, best_decoder = train_model(encoder, decoder, criterion, optimizer, scheduler, num_epochs=num_epochs)
    print("===== Evaluation Start =====")
    eval_model(best_encoder, best_decoder)
