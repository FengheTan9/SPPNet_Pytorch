import torch.nn as nn
import torch
import math

class Spatial_pyramid_pool(nn.Module):

    def __init__(self, previous_conv_size=13, out_pool_size=[6, 3, 2, 1], batch_size=256):
        super(Spatial_pyramid_pool, self).__init__()
        self.previous_conv_size = previous_conv_size
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size

    def forward(self, x):
        spp = None
        for pool_count in range(len(self.out_pool_size)):
            size = int(math.ceil(self.previous_conv_size / self.out_pool_size[pool_count]))
            stride = int(math.floor(self.previous_conv_size / self.out_pool_size[pool_count]))
            if self.out_pool_size[pool_count]==6 and self.previous_conv_size==10:
                maxpool = nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
            else:
                maxpool = nn.MaxPool2d(kernel_size=size, stride=stride)
            temp = maxpool(x)

            if pool_count == 0:
                spp = temp.view(self.batch_size, -1)
            else:
                spp = torch.cat((spp, temp.view(self.batch_size, -1)), 1)
        return spp
