import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import learning.viz as viz
from collections import namedtuple
from util import util
import os
torch.backends.cudnn.enabled = True

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')
name_lookup = {'Prismatic': 0, 'Revolute': 1}


def train_eval(args, hdim, batch_size, pviz, fname, data_fname):
    # Load data
    #data = parse_pickle_file(fname=args.data_fname)
    data = parse_pickle_file(fname=data_fname)
    train_set, val_set, test_set = setup_data_loaders(data=data,
                                                      batch_size=batch_size,
                                                      small_train=args.n_train)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[2, 12],
                   hdim=hdim,
                   im_h=53,  # 154,
                   im_w=115,  # 205,
                   kernel_size=3,
                   image_encoder=args.image_encoder)

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    if args.use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    # Add the graph to TensorBoard viz,
    k, x, q, im, y, _ = train_set.dataset[0]
    pol = torch.Tensor([name_lookup[k]])
    if args.use_cuda:
        x = x.cuda().unsqueeze(0)
        q = q.cuda().unsqueeze(0)
        im = im.cuda().unsqueeze(0)
        pol = pol.cuda()

    best_val = 1000
    # Training loop.
    vals = []
    for ex in range(1, args.n_epochs+1):
        train_losses = []
        net.train()
        for bx, (k, x, q, im, y, _) in enumerate(train_set):
            pol = name_lookup[k[0]]
            if args.use_cuda:
                x = x.cuda()
                q = q.cuda()
                im = im.cuda()
                y = y.cuda()
            optim.zero_grad()
            yhat = net.forward(pol, x, q, im)

            loss = loss_fn(yhat, y)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []
            net.eval()

            ys, yhats, types = [], [], []
            for bx, (k, x, q, im, y, _) in enumerate(val_set):
                pol = torch.Tensor([name_lookup[k[0]]])
                if args.use_cuda:
                    x = x.cuda()
                    q = q.cuda()
                    im = im.cuda()
                    y = y.cuda()

                yhat = net.forward(pol, x, q, im)

                loss = loss_fn(yhat, y)
                val_losses.append(loss.item())

                types += k
                if args.use_cuda:
                    y = y.cpu()
                    yhat = yhat.cpu()
                ys += y.numpy().tolist()
                yhats += yhat.detach().numpy().tolist()

            curr_val = np.mean(val_losses)
            vals += [[ex, curr_val]]
            print('[Epoch {}] - Validation Loss: {}'.format(ex, curr_val))
            if curr_val < best_val:
                best_val = curr_val
                best_epoch = ex

                # save model
                model_fname = fname+'epoch_'+str(best_epoch)
                full_path = model_fname+'.pt'
                torch.save(net.state_dict(), full_path)

                # save plot of prediction error
                if pviz:
                    viz.plot_y_yhat(ys, yhats, types, ex, fname, title='PolVis')
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
    parser.add_argument('--data-fname', type=str)
    parser.add_argument('--model-prefix', type=str, default='model')
    # if 0 then use all samples in dataset, else use ntrain number of samples
    parser.add_argument('--n-train', type=int, default=0)
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    parser.add_argument('--n-runs', type=int, default=1)
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
                    fname = args.model_prefix+'_nrun_'+str(n)
                    all_vals_epochs, best_epoch = train_eval(args, hdim, batch_size, False, fname)
                    es = [v[0] for v in all_vals_epochs]
                    vals = [v[1] for v in all_vals_epochs]
                    plot_val_error(es, vals, 'epoch', fname)
                    run_data += [RunData(hdim, batch_size, n, args.n_epochs, best_epoch, min(vals))]
        util.write_to_file(fname+'_results', run_data)
    elif args.mode == 'ntrain':
        vals = []
        #step = 5
        #ns = range(step, args.n_train+1, step)
        #for n in [50000]:
        for n_bbs in range(60,101,10):
            data_fname = 'active_datasets/active_' + str(n_bbs) + 'bb_200int.pickle'
            fname = 'all_models/'+args.model_prefix+'_nbbs_'+str(n_bbs)+'/'
            os.makedirs(fname)
            all_vals_epochs, best_epoch = train_eval(args, args.hdim, args.batch_size, False, fname, data_fname)
            best_val = min([ve[1] for ve in all_vals_epochs])
            vals.append(best_val)
        #plot_val_error(ns, vals, 'n train', args.model_prefix+'ntrain')
