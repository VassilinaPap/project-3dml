import torch
import torch.nn as nn
import torchvision.models as models

class MVCNN(nn.Module):
    def __init__(self, n_classes=13, encoder='mobilenet_v3', num_views=3, flag_rec = True, flag_multibranch = True):
        super(MVCNN, self).__init__()

        self.n_classes = n_classes
        self.num_views = num_views
        self.encoder = encoder
        self.flag_rec = flag_rec
        self.flag_multibranch = flag_multibranch
        self.decoder = None
        self.fuse_cl_rec = None
        self.decoder_volume = None

        if self.encoder == 'mobilenet_v3':
            self.net = models.mobilenet_v3_large(pretrained=True)

            self.net_1 = nn.Sequential(*list(self.net.children())[:-1])     #[B, 960, 137, 137]

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


        self.fuse_cl_rec = nn.Sequential(
            nn.Conv1d(in_channels=960+self.n_classes, out_channels=960, kernel_size=1, padding=0),
            nn.BatchNorm1d(960),
            nn.ReLU()
        )

        self.diff_fuse = nn.ModuleList()
        for i in range(num_views):
            self.diff_fuse.append(
                nn.Sequential(
            nn.Conv1d(in_channels=960+self.n_classes, out_channels=960, kernel_size=1, padding=0),
            nn.BatchNorm1d(960),
            nn.ReLU()
            )
        )

        self.decoder = nn.Sequential(
            # Layer 1: out [B, 64, 4, 4, 4]
            nn.ConvTranspose3d(in_channels=120, out_channels=64, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Layer 2: out [B, 32, 8, 8, 8]
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Layer 3: out [B, 16, 16, 16, 16]
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Layer 4: out [B, 8, 32, 32, 32]
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.decoder_volume = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid() # to deal with problem as a binary classification task
        )

        self.diff_branches = nn.ModuleList()
        for i in range(num_views):
            self.diff_branches.append(
                nn.Sequential(
            # Layer 1: out [B, 64, 4, 4, 4]
            nn.ConvTranspose3d(in_channels=120, out_channels=64, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            # Layer 2: out [B, 32, 8, 8, 8]
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            # Layer 3: out [B, 16, 16, 16, 16]
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            # Layer 4: out [B, 8, 32, 32, 32]
            nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=4, stride=2, bias=False, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )
            )

        self.diff_volume = nn.ModuleList()
        for i in range(num_views):
            self.diff_volume.append(
                nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid() # to deal with problem as a binary classification task
        )

            )


    def forward(self, x):
        # x [batch, views, 3, 137, 137] #
        batch_size = x.shape[0]
        x = x.transpose(0, 1) # [views, batch, 3, 137, 137]
        #print(x.shape)          

        feature_list = []
        for view in x: # [batch, 3, 137, 137]
            view_features = self.net_1(view) # [batch, 960, 1, 1]
            #print(view_features.shape)
            view_features = view_features.view(view_features.shape[0], -1) # [batch, 960]
            #print(view_features.shape)
            feature_list.append(view_features)

        max_features = feature_list[0]
        for view_features in feature_list[1:]:
            max_features = torch.max(max_features, view_features)# [batch, 960]
        #print(max_features.shape)

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
        """

        #classification branch
        cl_ret = self.net_2(max_features) # [batch, classes]
        #print(cl_ret)
        if not self.flag_rec:
            return cl_ret, None
        
        #reconstruction branch

        if not self.flag_multibranch: 

            max_features = max_features[:, :, None ]
            #print(max_features.shape)

            #print(cl_ret.shape)
            cl_ret3d = cl_ret[:, :, None ]

            #print(cl_ret.shape)
            #print(cl_ret3d.shape)
            features = self.fuse_cl_rec(torch.cat((max_features, cl_ret3d), dim=1))

            features = features.view(batch_size, -1, 2, 2, 2) # [B, C, 2, 2, 2]

            decoded_features = self.decoder(features)
            #print(decoded_features.shape)

            generated_volume = self.decoder_volume(decoded_features) # [B, 1, 32, 32, 32]
            rec_ret = generated_volume.squeeze()

        else: #multibranch
        
            cl_ret3d = cl_ret[:, :, None ]

            for index, feature in enumerate(feature_list):
                feature = feature[:,:,None]
                feature_list[index] = self.diff_fuse[index](torch.cat((feature, cl_ret3d), dim=1))


            generated_volume_list = []
            for idx, view_features in enumerate(feature_list):
                view_features = view_features.view(batch_size, -1, 2, 2, 2) # [B, C, 2, 2, 2]

                decoded_features = self.diff_branches[idx](view_features) # [B, 8, 32, 32, 32]
                
                generated_volume = self.diff_volume[idx](decoded_features) # [B, 1, 32, 32, 32]

                #raw_decoded_features = torch.cat((raw_decoded_features, generated_volume), dim=1) # [B, 9, 32, 32, 32]

                generated_volume_list.append(generated_volume.squeeze())
                #raw_decoded_features_list.append(raw_decoded_features)

            #rec_ret = torch.mean(torch.stack(generated_volume_list, dim=0), dim=0) # average pooling
            rec_ret = generated_volume_list[0] # max pooling
            for view_features in generated_volume_list[1:]:
                rec_ret = torch.max(rec_ret, view_features)
        

        return cl_ret, rec_ret
