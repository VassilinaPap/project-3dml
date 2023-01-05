import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class MVCNN(nn.Module):

    def __init__(self, nclasses=13, cnn_name='mobilenet_v3', num_views=3):
        super(MVCNN, self).__init__()

        self.nclasses = nclasses
        self.num_views = num_views

        if cnn_name=='mobilenet_v3':
            self.net = models.mobilenet_v3_large(pretrained=True)
            self.net_1 = nn.Sequential(*list(self.net.children())[:-1])
            self.net_2 = nn.Sequential(nn.Linear(960,480), nn.ReLU(True),
                    nn.Linear(480, self.nclasses))
        elif cnn_name=='resnet18':
            self.net = models.resnet18(pretrained=True)
            self.net_1 = nn.Sequential(*list(self.net.children())[:-4])
            self.net_2 = nn.Linear(512,self.nclasses)

    def forward(self, x):
        # [batch=2, views=4, 3, 137, 137] #
        x = x.transpose(0, 1)
        # [views=4, batch=2, 3, 137, 137] #
        feature_list = []
        for view in x:
            view_features = self.net_1(view)
            view_features = view_features.view(view_features.shape[0], -1)
            feature_list.append(view_features)

        # 4 -> [2, 960] #
        # feature_list  #
        max_features = feature_list[0]
        for view_features in feature_list[1:]:
            max_features = torch.max(max_features, view_features)

        # [2, 960] #
        ret = self.net_2(max_features)

        return ret
