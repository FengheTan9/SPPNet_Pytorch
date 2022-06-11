import torch.nn as nn
from src.spatial_pyramid_pooling import Spatial_pyramid_pool


class SPPNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000, include_top=True, batch=256, gpu_num=1):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(SPPNet, self).__init__()
        # input size should be : (b x 3 x 224 x 224)

        self.batch = int(batch/gpu_num)
        self.before_spp_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=(2, 2), padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,alpha=1.0,beta=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, (5, 5), padding=2, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1.0, beta=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, (3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.after_spp_net = nn.Sequential(
            nn.Linear(in_features=(6 * 6 + 3 * 3 + 2 * 2 + 1 * 1) * 256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.58),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.58),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.include_top = include_top
        self.spp_pool_224 = Spatial_pyramid_pool(13, [6, 3, 2, 1], self.batch)
        self.spp_pool_180 = Spatial_pyramid_pool(10, [6, 3, 2, 1], self.batch)

    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        N, C, H, W = x.size()

        x = self.before_spp_net(x)

        if H == 224:
            x = self.spp_pool_224(x)
        else:
            x = self.spp_pool_180(x)
        if not self.include_top:
            return x

        x = self.after_spp_net(x)
        return x
