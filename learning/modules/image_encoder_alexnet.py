import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
from torchvision import transforms


class ImageEncoder(nn.Module):

    def __init__(self, hdim, H, W, kernel_size=3):
        """

        :param hdim: Number of hidden units to use in each layer.
        :param H: The initial height of an image.
        :param W: The initial width of an image.
        :param kernel_size: The kernel_size of the convolution.
        """
        super(ImageEncoder, self).__init__()

        anet = alexnet(pretrained=True)
        f = list(anet.features)[:3]
        print(f)
        self.features = nn.Sequential(*f)
        for p in self.features.parameters():
            p.requires_grad = True

        self.conv21 = nn.Conv2d(in_channels=64,
                                out_channels=hdim,
                                kernel_size=kernel_size,
                                padding=1)
        self.conv22 = nn.Conv2d(in_channels=hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size,
                                padding=1)
        self.conv31 = nn.Conv2d(in_channels=hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size,
                                padding=1)
        self.conv32 = nn.Conv2d(in_channels=hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size,
                                padding=1)

        # I am currently just running the network to see what this size should be.
        self.lin_input = 24
        self.fc1 = nn.Linear(self.lin_input, hdim)
        self.fc2 = nn.Linear(hdim, hdim)

    def forward(self, img):
        x = self.features(img)
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, self.lin_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



