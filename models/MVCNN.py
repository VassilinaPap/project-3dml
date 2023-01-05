import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models


# mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
# std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

# def flip(x, dim):
#     xsize = x.size()
#     dim = x.dim() + dim if dim < 0 else dim
#     x = x.view(-1, *xsize[dim:])
#     x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
#                       -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
#     return x.view(xsize)


# class SVCNN(Model):

#     def __init__(self, name, nclasses=13, pretraining=True, cnn_name='vgg11'):
#         super(SVCNN, self).__init__(name)

#         self.classnames=["airplane",     "bench",     "cabinet",     "car",     "chair",     "display",     "lamp",     "loudspeaker",     "rifle",
#         "sofa",    "table",     "telephone",     "watercraft"]

#         self.nclasses = nclasses
#         self.pretraining = pretraining
#         self.cnn_name = cnn_name
#         self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
#         self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        
#         if self.cnn_name == 'resnet18':
#             self.net = models.resnet18(pretrained=self.pretraining)
#             self.net.fc = nn.Linear(512,40)
#         elif self.cnn_name == 'mobilenet_v3_large':
#             self.net = models.mobilenet_v3_large(weights='IMAGENET1K_V2')
#             self.net.fc = nn.Linear(1280,40)
#         elif self.cnn_name == 'alexnet':
#             self.net_1 = models.alexnet(pretrained=self.pretraining).features
#             self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
#         elif self.cnn_name == 'vgg11':
#             self.net_1 = models.vgg11(pretrained=self.pretraining).features
#             self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
#         elif self.cnn_name == 'vgg16':
#             self.net_1 = models.vgg16(pretrained=self.pretraining).features
#             self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
            
#             self.net_2._modules['6'] = nn.Linear(4096,40)

#     def forward(self, x):
#         if self.use_resnet:
#             return self.net(x)
#         else:
#             y = self.net_1(x)
#             return self.net_2(y.view(y.shape[0],-1))


class MVCNN(nn.Module):

    def __init__(self, nclasses=40, cnn_name='mobilenet_v3', num_views=3):
        super(MVCNN, self).__init__()

        # self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
        #                  'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
        #                  'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
        #                  'person','piano','plant','radio','range_hood','sink','sofa','stairs',
        #                  'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        # self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        # self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()



        if cnn_name=='mobilenet_v3':
            self.net = models.mobilenet_v3_large(pretrained=True)
            self.net_1 = nn.Sequential(*list(self.net.children())[:-1])
            self.net_2 = nn.Linear(960,13)
        elif cnn_name=='resnet18':
            self.net = models.resnet18(pretrained=True)
            self.net_1 = nn.Sequential(*list(self.net.children())[:-1])
            self.net_2 = nn.Linear(512,13)


    def forward(self, x):
        # [batch=2, views=4, 3, 137, 137] #
        x = x.transpose(0, 1)
        # [views=4, batch=2, 3, 137, 137] #
        feature_list = []
        for view in x:
            view_features = self.net_1(view)
            view_features = view_features.view(view_features.shape[0], -1)
            feature_list.append(view_features)

        max_features = feature_list[0]
        for view_features in feature_list[0:]:
            max_features = torch.max(max_features, view_features)

        ret = self.net_2(max_features)

        return ret
