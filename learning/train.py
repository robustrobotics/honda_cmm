import argparse
import numpy as np
import torch
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders
import learning.viz as viz


def train_eval(args, n_train, data_file_name, model_file_name, pviz, use_cuda):
    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname=data_file_name,
                                                      batch_size=args.batch_size,
                                                      small_train=n_train)

    # Setup Model (TODO: Update the correct policy dims)
    if args.model == 'pol':
        net = NNPol(policy_names=['Prismatic', 'Revolute'],
                    policy_dims=[9, 12],
                    hdim=args.hdim)
    else:
        net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                       policy_dims=[9, 12],
                       hdim=args.hdim,
                       im_h=53,  # 154,
                       im_w=115,  # 205,
                       kernel_size=3)
    if use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    best_val = 1000
    # Training loop.
    vals = []
    for ex in range(1, args.n_epochs+1):
        train_losses = []
        net.train()
        for bx, (k, x, q, im, y) in enumerate(train_set):
            if use_cuda:
                x = x.cuda()
                q = q.cuda()
                im = im.cuda()
                y = y.cuda()
            optim.zero_grad()
            yhat = net.forward(k[0], x, q, im)

            loss = loss_fn(yhat, y)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            val_losses = []
            net.eval()

            ys, yhats, types = [], [], []
            for bx, (k, x, q, im, y) in enumerate(val_set):
                if use_cuda:
                    x = x.cuda()
                    q = q.cuda()
                    im = im.cuda()
                    y = y.cuda()

                yhat = net.forward(k[0], x, q, im)
                loss = loss_fn(yhat, y)

                val_losses.append(loss.item())

                types += k
                if use_cuda:
                    y = y.cpu()
                    yhat = yhat.cpu()
                ys += y.numpy().tolist()
                yhats += yhat.detach().numpy().tolist()

            if pviz:
                viz.plot_y_yhat(ys, yhats, types, ex, title='PolVis')

            print('[Epoch {}] - Validation Loss: {}'.format(ex, np.mean(val_losses)))
            if np.mean(val_losses) < best_val:
                best_val = np.mean(val_losses)
            file_name = 'data/models/'+model_file_name+'_'+str(n_train)+'.pt'
            torch.save(net.state_dict(), file_name)
            vals += [np.mean(val_losses)]
    return best_val, vals

def plot_val_error(ns, vals, type):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.xlabel(type)
        plt.ylabel('Val MSE')

        plt.plot(ns, vals)
        plt.savefig('val_error.png', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=16, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--mode', choices=['ntrain', 'normal'], default='normal')
    parser.add_argument('--model', choices=['pol', 'polvis'], default='polvis')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-fname', type=str, required=True) # ending in .pickle in data/datasets
    parser.add_argument('--model-fname', type=str, required=True) # ending in .pt
    parser.add_argument('--ntrain', type=int, default=0)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    data_file_name = args.data_fname

    if args.mode == 'normal':
        best_val, vals = train_eval(args, args.ntrain, data_file_name, args.model_fname, True, args.use_cuda)
        xs = [x for x in range(args.val_freq,args.n_epochs+1,args.val_freq)]
        plot_val_error(xs, vals, 'epoch')
    elif args.mode == 'ntrain':
        vals = []
        step = 500
        ns = range(step, args.ntrain+1, step)
        try:
            for n in ns:
                best_val = train_eval(args, n, data_file_name, args.model_fname, False, args.use_cuda)
                vals.append(best_val)
                print(n, best_val)
            plot_val_error(ns, vals, 'n train')
        except:
            plot_val_error(ns, vals, 'n train')
