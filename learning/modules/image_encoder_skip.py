import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):

    def __init__(self, hdim, H, W, kernel_size=3):
        """

        :param hdim: Number of hidden units to use in each layer.
        :param H: The initial height of an image.
        :param W: The initial width of an image.
        :param kernel_size: The kernel_size of the convolution.
        """
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3+hdim,
                               out_channels=hdim,
                               kernel_size=kernel_size)
        self.conv21 = nn.Conv2d(in_channels=2*hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size)
        self.conv22 = nn.Conv2d(in_channels=2*hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size)
        self.conv31 = nn.Conv2d(in_channels=2*hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size)
        self.conv32 = nn.Conv2d(in_channels=2*hdim,
                                out_channels=hdim,
                                kernel_size=kernel_size)
        self.conv4 = nn.Conv2d(in_channels=2*hdim,
                               out_channels=hdim,
                               kernel_size=kernel_size)

        # Reference: https://pytorch.org/docs/stable/nn.html#maxpool2d
        for k in range(0, 4):
            # Output size of kth conv layer
            H = H - kernel_size + 1
            W = W - kernel_size + 1

            # Output size of kth pooling layer
            H = int((H - (2 - 1) - 1)/2.0 + 1)
            W = int((W - (2 - 1) - 1)/2.0 + 1)

        # I am currently just running the network to see what this size should be.
        self.lin_input = hdim*3*11  # H*W*hdim
        self.fc1 = nn.Linear(self.lin_input, hdim)
        self.fc2 = nn.Linear(hdim, hdim)

    def forward(self, img, pol):
        pol = pol.unsqueeze(-1).unsqueeze(-1)

        p = pol.expand(pol.shape[0], pol.shape[1], img.shape[2], img.shape[3])
        img = torch.cat([p, img], dim=1)
        x = F.relu(self.conv1(img))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        p = pol.expand(pol.shape[0], pol.shape[1], x.shape[2], x.shape[3])
        x = torch.cat([x, p], dim=1)
        x = F.relu(self.conv21(x))
        p = pol.expand(pol.shape[0], pol.shape[1], x.shape[2], x.shape[3])
        x = torch.cat([x, p], dim=1)
        x = F.relu(self.conv22(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        p = pol.expand(pol.shape[0], pol.shape[1], x.shape[2], x.shape[3])
        x = torch.cat([x, p], dim=1)
        x = F.relu(self.conv31(x))
        p = pol.expand(pol.shape[0], pol.shape[1], x.shape[2], x.shape[3])
        x = torch.cat([x, p], dim=1)
        x = F.relu(self.conv32(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, self.lin_input)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



