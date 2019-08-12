import torch.nn as nn
import numpy as np
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as ImageEncoder


class Autoencoder(nn.Module):

    def __init__(self, hdim, recon_shape, n_features):
        """

        :param hdim: Hidden dimensions to use in the neural network.
        :param recon_shape: Shape of the downsampled image.
        :param n_features: How many features the spatial autoencoder should have.
        """
        super(Autoencoder, self).__init__()
        self.encoder = ImageEncoder(hdim=hdim,
                                    kernel_size=7,
                                    n_features=n_features)
        self.decoder = nn.Linear(hdim*2, np.prod(recon_shape))
        self.recon_shape = recon_shape

    def forward(self, img):
        feat = self.encoder(img)
        recon = self.decoder(feat)
        return recon.view(-1,
                          self.recon_shape[0],
                          self.recon_shape[1],
                          self.recon_shape[2])
