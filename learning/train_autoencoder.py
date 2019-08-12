import argparse
import numpy as np
import torch
from torchvision.transforms import Resize
from learning.dataloaders import setup_data_loaders
from learning.models.autoencoder import Autoencoder
from collections import namedtuple
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter

import sys

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')
name_lookup = {'Prismatic': 0, 'Revolute': 1}


def train_eval(args, pviz, fname):
    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname=args.data_fname,
                                                      batch_size=args.batch_size,
                                                      small_train=0)
    recon_size = train_set.dataset.downsampled_images[0].shape
    # Setup Model
    net = Autoencoder(hdim=args.hdim,
                      recon_shape=recon_size,
                      n_features=args.n_features)

    if args.use_cuda:
        net = net.cuda()

    writer = SummaryWriter()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    best_val = 1000
    best_epoch = -1
    # Training loop.
    vals = []
    for ex in range(1, args.n_epochs+1):
        train_losses = []
        net.train()
        for bx, (_, _, _, im, _, y_im) in enumerate(train_set):
            if args.use_cuda:
                im = im.cuda()
                y_im = y_im.cuda()

            optim.zero_grad()
            yhat = net.forward(im)

            # TODO: Create a down-sampled image as y.

            loss = loss_fn(yhat, y_im)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []
            net.eval()

            for bx, (_, _, _, im, _, y_im) in enumerate(val_set):
                if args.use_cuda:
                    im = im.cuda()
                    y_im = y_im.cuda()

                yhat = net.forward(im)
                if bx == 0:
                    for kx in range(0, yhat.shape[0]):
                        writer.add_image('recon_%d' % kx, yhat[kx, 0, :, :], dataformats='HW', global_step=ex)

                loss = loss_fn(yhat, y_im)
                val_losses.append(loss.item())

            curr_val = np.mean(val_losses)
            vals += [[ex, curr_val]]
            print('[Epoch {}] - Validation Loss: {}'.format(ex, curr_val))
            if curr_val < best_val:
                best_val = curr_val
                best_epoch = ex

                # save model
                model_fname = fname+'_epoch_'+str(best_epoch)
                full_path = 'data/models/'+model_fname+'.pt'
                torch.save(net.state_dict(), full_path)

                # save plot of prediction error
                if pviz:
                    if args.use_cuda:
                        y_im = y_im.cpu()
                        yhat = yhat.cpu()
                    imshow(y_im, yhat)

    return vals, best_epoch


def imshow(img, recon):
    n, c, h, w = img.shape
    print(img.shape)
    # img = img.numpy()
    recon = recon.detach()

    fig, axes = plt.subplots(2, 1)

    # for ix in range(5):
    #     print(ix)
    #     print(np.transpose(img[ix, :, :, :], (1, 2, 0)).shape)
    #     axes[ix][0].imshow(img)
    #     axes[ix][1].imshow(recon)

    imgs = torchvision.utils.make_grid(img[0:10]).numpy()
    recons = torchvision.utils.make_grid(recon[0:10]).numpy()
    imgs = np.transpose(imgs, (1, 2, 0))
    recons = np.transpose(recons, (1, 2, 0))
    axes[0].imshow(imgs)
    axes[1].imshow(recons)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-features', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--data-fname', type=str, required=True)
    parser.add_argument('--model-prefix', type=str, default='model')
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    args = parser.parse_args()

    fname = args.model_prefix + '_autoencoder'
    train_eval(args, False, fname)

