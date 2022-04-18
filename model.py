from __future__ import print_function

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()

        self.k = k

        # Each layer has batchnorm and relu on it
        #TODO
        # layer 1: k -> 64

        #TODO
        # layer 2:  64 -> 128

        #TODO
        # layer 3: 128 -> 1024

        #TODO
        # fc 1024 -> 512

        #TODO
        # fc 512 -> 256

        #TODO
        # fc 256 -> k*k (no batchnorm, no relu)

        #TODO
        # ReLU activationfunction


    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # apply layer 1

        #TODO
        # apply layer 2

        #TODO
        # apply layer 3

        #TODO
        # do maxpooling and flatten


        #TODO
        # apply fc layer 1

        #TODO
        # apply fc layer 2

        #TODO
        # apply fc layer 3

        #TODO
        #reshape output to a b*k*k tensor

        #TODO
        # define an identity matrix to add to the output. This will help with the stability of the results since we want our transformations to be close to identity


        #TODO
        # return output



class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = True):
        super(PointNetfeat, self).__init__()

        self.feature_transform= feature_transform
        self.global_feat = global_feat

        #TODO
        # Use TNet to apply transformation on input and multiply the input points with the transformation

        #TODO
        # layer 1:3 -> 64

        #TODO
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)

        #TODO
        # layer2: 64 -> 128

        #TODO
        # layer 3: 128 -> 1024 (no relu)

        #TODO
        # ReLU activation



    def forward(self, x):
        batch_size, _, num_points = x.shape
        #TODO
        # input transformation, you will need to return the transformation matrix as you will need it for the regularization loss

        #TODO
        # apply layer 1

        #TODO
        # feature transformation, you will need to return the transformation matrix as you will need it for the regularization loss

        #TODO
        # apply layer 2

        #TODO
        # apply layer 3

        #TODO
        # apply maxpooling


        #TODO
        # return output, input transformation matrix, feature transformation matrix
        if self.global_feat: # This shows if we're doing classification or segmentation
            if self.feature_transform:

        else:
            if self.feature_transform:


class PointNetCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_classes = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        #TODO
        # get global features + point features from PointNetfeat

        #TODO
        # layer 1: 1088 -> 512

        #TODO
        # layer 2: 512 -> 256

        #TODO
        # layer 3: 256 -> 128

        #TODO
        # layer 4:  128 -> k (no ru and batch norm)

        #TODO
        # ReLU activation


    
    def forward(self, x):
        #TODO
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)


        #TODO
        # apply layer 1

        #TODO
        # apply layer 2

        #TODO
        # apply layer 3

        #TODO
        # apply layer 4

        #TODO
        # apply log-softmax


def feature_transform_regularizer(trans):

    batch_size, feature_size, _ = trans.shape
    #TODO
    # compute I - AA^t

    #TODO
    # compute norm

    #TODO
    # compute mean norms and return



if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(num_classes = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(num_classes = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())

