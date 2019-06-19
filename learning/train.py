import argparse
import numpy as np
import torch
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=128, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=1)
    args = parser.parse_args()

    # Load data
    train_set, val_set, test_set = setup_data_loaders(batch_size=args.batch_size)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPol(policy_names=['prismatic', 'revolute'],
                policy_dims=[10, 10])
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    # Training loop.
    for ex in range(args.n_epochs):
        train_losses = []
        for bx, (x, y) in enumerate(train_set):
            optim.zero_grad()

            yhat = net.forward(x)
            loss = loss_fn(yhat, y)
            loss.backward()

            optim.step()

            train_losses.append(loss)

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []
            for bx, (x, y) in enumerate(val_set):
                yhat = net.forward(x)
                loss = loss_fn(yhat, y)
                val_losses.append(loss)

            print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(val_losses)))


