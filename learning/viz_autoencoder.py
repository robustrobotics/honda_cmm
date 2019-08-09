import matplotlib.pyplot as plt
import numpy as np
import torch
from learning.dataloaders import setup_data_loaders
from learning.models.autoencoder import Autoencoder
from learning.train_autoencoder import imshow


def view_features_maps():
    pass


def view_recons(data, net):
    for bx, (_, _, _, im, _, y_im) in enumerate(data):
        im = im.cuda()
        y_im = y_im.cuda()

        yhat = net.forward(im)
        imshow(y_im.cpu(), yhat.cpu())
        break


if __name__ == '__main__':
    train_set, val_set, test_set = setup_data_loaders(fname='prism_rand05_10k.pickle',
                                                      batch_size=10,
                                                      small_train=0)

    recon_size = train_set.dataset.downsampled_images[0].shape
    # Setup Model
    net = Autoencoder(hdim=16,
                      recon_shape=recon_size).cuda()
    device = torch.device('cuda')
    net.load_state_dict(torch.load('data/models/model_autoencoder_epoch_170.pt', map_location=device))
    net.eval()

    view_recons(test_set, net)
