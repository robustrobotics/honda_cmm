import argparse
import numpy as np
import torch
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders
import learning.viz as viz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=16, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=1)
    args = parser.parse_args()

    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname='6_20_19_2340.pickle',
                                                      batch_size=args.batch_size)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPol(policy_names=['Prismatic', 'Revolute'],
                policy_dims=[11, 17],
                hdim=args.hdim).cuda()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    # Training loop.
    for ex in range(args.n_epochs):
        train_losses = []
        for bx, (k, x, q, y) in enumerate(train_set):
            x = x.cuda()
            q = q.cuda()
            y = y.cuda()

            optim.zero_grad()
            qs = torch.zeros(x.shape[0], 1)
            yhat = net.forward(k[0], x, q)

            loss = loss_fn(yhat, y)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []

            ys, yhats, types = [], [], []
            for bx, (k, x, q, y) in enumerate(val_set):
                x = x.cuda()
                q = q.cuda()
                y = y.cuda()

                qs = torch.zeros(x.shape[0], 1)
                yhat = net.forward(k[0], x, q)
                loss = loss_fn(yhat, y)

                val_losses.append(loss.item())

                types += k
                ys += y.cpu().numpy().tolist()
                yhats += yhat.cpu().detach().numpy().tolist()

            viz.plot_y_yhat(ys, yhats, types)

            print('[Epoch {}] - Validation Loss: {}'.format(ex, np.mean(val_losses)))


