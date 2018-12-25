import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.C3D import C3D
from models.resnext import resnext101
from models.resnet import resnet101
from collections import OrderedDict
# from EmotiW2018.VideoBased.Feature_wise_Fusion.C3D import C3D

WEIGHT_C3D_PATH = '/home/dmsl/nas/HDD_server6/EmotiW/data/C3D/C3D_keras_to_pytorch.pkl'
WEIGHT_RESNEXT3D_PATH = '/home/dmsl/nas/HDD_server6/EmotiW/data/Resnext3D/resnext-101-kinetics.pth'
WEIGHT_RESNET3D_PATH = '/home/dmsl/nas/HDD_server6/EmotiW/data/Resnet3D/resnet-101-kinetics.pth'

def removeModule(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict

class EncoderC3D(nn.Module):
    def __init__(self, embed_size):
        super(EncoderC3D, self).__init__()
        base_model = C3D().cuda()
        base_model.load_state_dict(torch.load(WEIGHT_C3D_PATH))

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
        self.bn = nn.BatchNorm1d(embed_size) # in case of ck+, useful
        self.classification = nn.Linear(4096, 7)


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
        h_ = self.fc8(h)
        feature = self.bn(h_)
        out_class = self.classification(h)

        return feature, out_class


class EncoderResnext3D(nn.Module):
    def __init__(self, embed_size):
        super(EncoderResnext3D, self).__init__()
        model_data = torch.load(WEIGHT_RESNEXT3D_PATH)
        model_state_dict = model_data['state_dict']
        base_model = resnext101(num_classes=400, shortcut_type='B', sample_size=112, sample_duration=16)
        base_model.load_state_dict(removeModule(model_state_dict))

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.fc6 = nn.Linear(2048, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(embed_size)
        self.classification = nn.Linear(2048, 7)


    def forward(self, x):
        # TODO : freeze c3d network using torch.no_grad()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        h = self.relu(self.fc6(x))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h_ = self.fc8(h)
        feature = self.bn(h_)
        out_class = self.classification(h)

        return feature, out_class

class EncoderResnet3D(nn.Module):
    def __init__(self, embed_size):
        super(EncoderResnet3D, self).__init__()
        model_data = torch.load(WEIGHT_RESNET3D_PATH)
        model_state_dict = model_data['state_dict']
        base_model = resnet101(num_classes=400, shortcut_type='B', sample_size=112, sample_duration=16)
        base_model.load_state_dict(removeModule(model_state_dict))

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.fc6 = nn.Linear(2048, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, embed_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(embed_size) # in case of CK+, useful
        self.classification = nn.Linear(2048, 7)


    def forward(self, x):
        # TODO : freeze c3d network using torch.no_grad()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        h = self.relu(self.fc6(x))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        h_ = self.fc8(h)
        feature = self.bn(h_)
        out_class = self.classification(h)

        return feature, out_class