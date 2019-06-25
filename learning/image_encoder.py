import torch
import torch.nn as nn


class ImageEncoder(nn.Module):

    def __init__(self, hdim, H, W, kernel_size=5):
        """

        :param hdim: Number of hidden units to use in each layer.
        :param H: The initial height of an image.
        :param W: The initial width of an image.
        :param kernel_size: The kernel_size of the convolution.
        """
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=hdim,
                               kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=4,
                                 stride=4)
        self.conv2 = nn.Conv2d(in_channels=hdim,
                               out_channels=hdim,
                               kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=hdim,
                               out_channels=hdim,
                               kernel_size=kernel_size)

        # Reference: https://pytorch.org/docs/stable/nn.html#maxpool2d
        for k in range(0, 3):
            # Output size of kth conv layer
            H = H - kernel_size + 1
            W = W - kernel_size + 1

            # Output size of kth pooling layer
            H = int((H - (4 - 1) - 1)/4.0 + 1)
            W = int((W - (4 - 1) - 1)/4.0 + 1)

        self.lin_input = H*W*hdim
        print('lin_input:', self.lin_input)
        self.fc1 = nn.Linear(self.lin_input, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.RELU = nn.ReLU()

    def forward(self, img):
        x = self.RELU(self.conv1(img))
        x = self.pool(x)
        x = self.RELU(self.conv2(x))
        x = self.pool(x)
        x = self.RELU(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, self.lin_input)
        x = self.RELU(self.fc1(x))
        x = self.fc2(x)
        return x
