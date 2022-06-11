import torch.nn as nn

class ZFNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self, num_classes=1000, batch=256, gpu_num=1):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super(ZFNet, self).__init__()
        # input size should be : (b x 3 x 224 x 224)
        self.batch = int(batch/gpu_num)
        self.zfnet = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.65),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.65),
            nn.Linear(in_features=4096, out_features=num_classes)
        )


    def forward(self, x):
        """
        Pass the input through the net.
        Args:
            x (Tensor): input tensor
        Returns:
            output (Tensor): output tensor
        """
        return self.zfnet(x)
