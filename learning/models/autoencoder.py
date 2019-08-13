import torch.nn as nn
import numpy as np
import torch
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as ImageEncoder


class Autoencoder(nn.Module):

    def __init__(self, hdim, recon_shape, n_features, decoder_type='basis'):
        """

        :param hdim: Hidden dimensions to use in the neural network.
        :param recon_shape: Shape of the downsampled image.
        :param n_features: How many features the spatial autoencoder should have.
        :param decoder_type: linear or basis
        """
        super(Autoencoder, self).__init__()
        self.encoder = ImageEncoder(hdim=hdim,
                                    kernel_size=7,
                                    n_features=n_features)
        if decoder_type == 'linear':
            self.decoder1 = nn.Linear(hdim*2, hdim*4)
            self.decoder2 = nn.Linear(hdim*4, np.prod(recon_shape))
        elif decoder_type == 'basis':
            self.decoder1 = nn.Linear(2*n_features, hdim)
            self.decoder2 = nn.Linear(hdim, 1)
        self.decoder_type = decoder_type
        self.recon_shape = recon_shape
        self.relu = nn.ReLU()

    def forward(self, img):
        if self.decoder_type == 'basis':
            img = img[:, :, 1:, 1:]
            bs, c, _, _ = img.shape
            h, w = self.recon_shape[1], self.recon_shape[2]
            pos_y, pos_x = torch.meshgrid(
                torch.linspace(-1., 1., h),
                torch.linspace(-1., 1., w))

            if next(self.parameters()).is_cuda:
                pos_y = pos_y.cuda()
                pos_x = pos_x.cuda()

            pos_x = torch.reshape(pos_x, [h * w]).unsqueeze(-1)
            pos_y = torch.reshape(pos_y, [h * w]).unsqueeze(-1)
            locations = torch.cat([pos_x, pos_y], dim=1).unsqueeze(0).expand(bs, -1, -1)

            feat, points, heatmaps = self.encoder(img)
            dim_f, dim_l = points.shape[1], locations.shape[1]
            L = locations.unsqueeze(2).expand(-1, -1, dim_f, -1)
            F = points.unsqueeze(1).expand(-1, dim_l, -1, -1)
            deltas = (L-F).reshape(bs, dim_l, dim_f*2)

            recon = self.decoder2(self.relu(self.decoder1(deltas)))
            return recon.view(-1,
                              self.recon_shape[0],
                              self.recon_shape[1],
                              self.recon_shape[2]), points, heatmaps
        elif self.decoder_type == 'linear':
            feat, points, heatmaps = self.encoder(img)
            recon = self.decoder2(self.relu(self.decoder1(feat)))
            return recon.view(-1,
                              self.recon_shape[0],
                              self.recon_shape[1],
                              self.recon_shape[2]), points, heatmaps
