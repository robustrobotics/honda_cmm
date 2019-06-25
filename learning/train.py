import argparse
import numpy as np
import torch
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders
import learning.viz as viz


def train_eval(args, n_train, fname, pviz):
    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname=fname,
                                                      batch_size=args.batch_size,
                                                      small_train=n_train)

    # Setup Model (TODO: Update the correct policy dims)
    if args.model == 'pol':
        net = NNPol(policy_names=['Prismatic', 'Revolute'],
                    policy_dims=[10, 14],
                    hdim=args.hdim).cuda()
    else:
        net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                       policy_dims=[10, 14],
                       hdim=args.hdim,
                       im_h=154,
                       im_w=205,
                       kernel_size=5).cuda()
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    best_val = 1000
    # Training loop.
    for ex in range(args.n_epochs):
        train_losses = []
        for bx, (k, x, q, im, y) in enumerate(train_set):
            x = x.cuda()
            q = q.cuda()
            im = im.cuda()
            y = y.cuda()

            optim.zero_grad()
            qs = torch.zeros(x.shape[0], 1)
            yhat = net.forward(k[0], x, q, im)

            loss = loss_fn(yhat, y)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        #print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []

            ys, yhats, types = [], [], []
            for bx, (k, x, q, im, y) in enumerate(val_set):
                x = x.cuda()
                q = q.cuda()
                im = im.cuda()
                y = y.cuda()

                qs = torch.zeros(x.shape[0], 1)
                yhat = net.forward(k[0], x, q, im)
                loss = loss_fn(yhat, y)

                val_losses.append(loss.item())

                types += k
                ys += y.cpu().numpy().tolist()
                yhats += yhat.cpu().detach().numpy().tolist()

            if pviz:
                viz.plot_y_yhat(ys, yhats, types, title='PolicyParams')

            print('[Epoch {}] - Validation Loss: {}'.format(ex, np.mean(val_losses)))
            if np.mean(val_losses) < best_val:
                best_val = np.mean(val_losses)

    return best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=16, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--mode', choices=['ntrain', 'normal'], default='normal')
    parser.add_argument('--model', choices=['pol', 'polvis'], default='pol')
    args = parser.parse_args()

    if args.mode == 'normal':
        train_eval(args, 0, 'clean_data.pickle', pviz=True)
    elif args.mode == 'ntrain':
        vals = []
        ns = range(100, 1001, 100)
        for n in ns:
            best_val = train_eval(args, n, 'clean_data.pickle', pviz=False)
            vals.append(best_val)
            print(n, best_val)

        import matplotlib.pyplot as plt
        plt.xlabel('n train')
        plt.ylabel('Val MSE')

        plt.plot(ns, vals)
        plt.show()
