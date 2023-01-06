import torch
import torch.nn as nn
import torchvision.models as models

class MVCNN(nn.Module):

    def __init__(self, n_classes=13, encoder='mobilenet_v3', num_views=3):
        super(MVCNN, self).__init__()

        self.n_classes = n_classes
        self.num_views = num_views
        self.encoder = encoder

        if self.encoder == 'mobilenet_v3':
            self.net = models.mobilenet_v3_large(pretrained=True)

            self.net_1 = nn.Sequential(*list(self.net.children())[:-1])

            self.net_2 = nn.Sequential(
                                nn.Linear(960, 480), nn.BatchNorm1d(480), nn.ReLU(True),
                                nn.Linear(480, self.n_classes)
                                )

        elif self.encoder == 'resnet18':

            self.net = models.resnet18(pretrained=True)

            self.net_1 = nn.Sequential(*list(self.net.children())[:-4])

            self.net_2 = nn.Sequential(
                                nn.Linear(512, self.n_classes)
                                )

    def forward(self, x):
        # x [batch, views, 3, 137, 137] #
        x = x.transpose(0, 1) # [views, batch, 3, 137, 137]

        """
        feature_list = []
        for view in x: # [batch, 3, 137, 137]
            view_features = self.net_1(view) # [batch, 960, 1, 1]
            view_features = view_features.view(view_features.shape[0], -1) # [batch, 960]
            feature_list.append(view_features)

        max_features = feature_list[0]
        for view_features in feature_list[1:]:
            max_features = torch.max(max_features, view_features)
        """

        v = x.shape[0]
        b = x.shape[1]
        c = x.shape[2]
        w = x.shape[3]
        h = x.shape[4]

        x = x.reshape(v * b, c, w, h)

        # Pass to encoder - example mobilenet dimensions #
        view_features = self.net_1(x) # [views * batch, 960]

        view_features = view_features.reshape(v, b, 960) # [views, batch, 960]

        # [batch, 960] #
        max_features = view_features[0]
        for view_features in view_features[1:]:
            max_features = torch.max(max_features, view_features)

        ret = self.net_2(max_features) # [batch, classes]

        return ret
