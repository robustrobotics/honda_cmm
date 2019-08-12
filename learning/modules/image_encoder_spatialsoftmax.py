import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class ImageEncoder(nn.Module):

    def __init__(self, hdim, n_features, kernel_size=7):
        """

        :param hdim: Number of hidden units to use in each layer.
        :param H: The initial height of an image.
        :param W: The initial width of an image.
        :param kernel_size: The kernel_size of the convolution.
        :param n_features: Number of features the spatial autoencoder should have.
        """
        super(ImageEncoder, self).__init__()
        self.pad = torch.nn.ReplicationPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=hdim,
                               kernel_size=kernel_size,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=hdim,
                               out_channels=n_features,
                               kernel_size=7,
                               padding=0)

        # I am currently just running the network to see what this size should be.
        #self.fc1 = nn.Linear(self.lin_input, hdim)
        self.scale = nn.Linear(2, 2)
        self.fc2 = nn.Linear(n_features*2, hdim*2)

        self.sm = nn.Softmax(dim=1)
        self.temp = nn.Parameter(torch.tensor(0.1))

    def forward(self, img):
        img = img[:, :, 1:, 1:]
        bs, c, h, w = img.shape
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1., 1., h),
            torch.linspace(-1., 1., w))
        if next(self.parameters()).is_cuda:
            pos_y = pos_y.cuda()
            pos_x = pos_x.cuda()

        pos_x = torch.reshape(pos_x, [h * w])
        pos_y = torch.reshape(pos_y, [h * w])
        img = self.pad(img)
        x = F.relu(self.conv1(img))
        x = self.pad(x)
        x = self.conv2(x)

        # Do a spatial softmax.
        bs, c, h, w = x.shape
        features = self.sm(x.view(bs*c, -1)/self.temp)
        
        # Get expected feature points.
        pfeatures = features.view([-1, h, w])

        expected_x = torch.sum(pos_x.unsqueeze(0).expand_as(features) * features,
                               dim=1,
                               keepdim=True)
        expected_y = torch.sum(pos_y.unsqueeze(0).expand_as(features) * features,
                               dim=1,
                               keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], dim=1)


        # imshow(torchvision.utils.make_grid(img), expected_xy.detach().numpy(), pfeatures.detach().numpy())
        # input()
        # sys.exit(0)

        expected_xy2 = self.scale(expected_xy)
        x = expected_xy2.view(-1, c*2)

        x = self.fc2(x)
        return x, expected_xy.view(bs, c, 2)

def imshow(img, points, maps):
    c, h, w = img.shape
    print(img.shape)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()

    fig, axes = plt.subplots(4, points.shape[0]//4+1)
    fig.subplots_adjust(hspace=0, wspace=0.1, top=0.9, bottom=0.1)

    axes = axes.flatten()
    axes[0].imshow(np.transpose(npimg, (1, 2, 0)))

    cmap = plt.get_cmap('viridis')

    print(points.shape)
    for ix in range(points.shape[0]):
        axes[0].scatter((points[ix, 0]+1)/2.0*w, (points[ix, 1]+1)/2.0*h, s=5, c=cmap(ix/points.shape[0]))


    for ix in range(0, maps.shape[0]):
        axes[ix+1].imshow(maps[ix, :, :])

    plt.show()
    input()


