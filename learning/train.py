import argparse
import numpy as np
import torch
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders
import learning.viz as viz
from collections import namedtuple
from util import util

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')

def train_eval(args, hdim, batch_size, pviz):
    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname=args.data_fname,
                                                      batch_size=batch_size,
                                                      small_train=args.n_train)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                       policy_dims=[2, 12],
                       hdim=hdim,
                       im_h=53,  # 154,
                       im_w=115,  # 205,
                       kernel_size=3)
    if args.use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss(reduction='none')
    optim = torch.optim.Adam(net.parameters())

    best_val = 1000
    # Training loop.
    vals = []
    for ex in range(1, args.n_epochs+1):
        train_losses = []
        net.train()
        for bx, (k, x, q, im, y) in enumerate(train_set):
            if args.use_cuda:
                x = x.cuda()
                q = q.cuda()
                im = im.cuda()
                y = y.cuda()
            optim.zero_grad()
            yhat = net.forward(k[0], x, q, im)

            loss_tensor = loss_fn(yhat, y)
            alpha = 1.0
            weights = 1.0 + alpha*y
            loss = torch.mean(weights*loss_tensor)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []
            net.eval()

            ys, yhats, types = [], [], []
            for bx, (k, x, q, im, y) in enumerate(val_set):
                if args.use_cuda:
                    x = x.cuda()
                    q = q.cuda()
                    im = im.cuda()
                    y = y.cuda()

                yhat = net.forward(k[0], x, q, im)
                loss = loss_fn(yhat, y)

                val_losses.append(loss.item())

                types += k
                if args.use_cuda:
                    y = y.cpu()
                    yhat = yhat.cpu()
                ys += y.numpy().tolist()
                yhats += yhat.detach().numpy().tolist()

            if pviz:
                viz.plot_y_yhat(ys, yhats, types, ex, title='PolVis')

            curr_val = np.mean(val_losses)
            vals += [[ex, curr_val]]
            print('[Epoch {}] - Validation Loss: {}'.format(ex, curr_val))
            if curr_val < best_val:
                best_val = curr_val
                best_epoch = ex
    model_fname = args.model_prefix+'_hdim_'+str(hdim)+'_bs_'+str(batch_size)+'_epoch_'+str(best_epoch)
    full_path = 'data/models/'+model_fname+'.pt'
    torch.save(net.state_dict(), full_path)
    return vals, best_epoch

def plot_val_error(ns, vals, type, fname=None, viz=False):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.xlabel(type)
        plt.ylabel('Val MSE')

        plt.plot(ns, vals)
        plt.savefig('val_error_'+fname+'.png', bbox_inches='tight')
        if viz:
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--mode', choices=['ntrain', 'normal'], default='normal')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-fname', type=str, required=True)
    parser.add_argument('--model-prefix', type=str, default='model')
    # if 0 then use all samples in dataset, else use ntrain number of samples
    parser.add_argument('--n-train', type=int, default=0)
    parser.add_argument('--n-runs', type=int, default=1)
    parser.add_argument('--results-fname', type=str, default='results')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # if hdim and batch_size given as args then use them, otherwise test a list of them
    if args.hdim:
        hdims = [args.hdim]
    else:
        hdims = [16, 32]
    if args.batch_size:
        batch_sizes = [args.batch_size]
    else:
        batch_sizes = [16, 32]

    if args.mode == 'normal':
        run_data = []
        for n in range(args.n_runs):
            for hdim in hdims:
                for batch_size in batch_sizes:
                    all_vals_epochs, best_epoch = train_eval(args, hdim, batch_size, False)
                    es = [v[0] for v in all_vals_epochs]
                    vals = [v[1] for v in all_vals_epochs]
                    fname = args.model_prefix+'_hdim_'+str(hdim)+'_bs_'+str(batch_size)+'_epoch_'+str(best_epoch)
                    plot_val_error(es, vals, 'epoch', fname)
                    run_data += [RunData(hdim, batch_size, n, args.n_epochs, best_epoch, min(vals))]
        util.write_to_file(args.results_fname, run_data)
    elif args.mode == 'ntrain':
        vals = []
        step = 500
        ns = range(step, args.n_train+1, step)
        for n in ns:
            all_vals_epochs, best_epoch = train_eval(args, args.hdim, args.batch_size, False)
            best_val = min([ve[1] for ve in all_vals_epochs])
            vals.append(best_eval)
        plot_val_error(ns, vals, 'n train', 'ntrain')
