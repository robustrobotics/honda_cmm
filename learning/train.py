import argparse
import numpy as np
import torch
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders
import learning.viz as viz
from torch.utils.tensorboard import SummaryWriter


def train_eval(args, n_train, data_file_name, model_file_name, pviz, use_cuda):
    # Load data
    train_set, val_set, test_set = setup_data_loaders(fname=data_file_name,
                                                      batch_size=args.batch_size,
                                                      small_train=n_train)

    # Setup Model (TODO: Update the correct policy dims)
    name_lookup = {'Prismatic': 0, 'Revolute': 1}
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

    # Add the graph to TensorBoard viz,
    writer = SummaryWriter()
    k, x, q, im, y = train_set.dataset[0]
    if use_cuda:
        x = x.cuda().unsqueeze(0)
        q = q.cuda().unsqueeze(0)
        im = im.cuda().unsqueeze(0)
    writer.add_graph(net, (torch.Tensor([0]).cuda(), x, q, im), operator_export_type="RAW")

    best_val = 1000
    # Training loop.
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
            yhat = net.forward(torch.Tensor([name_lookup[k[0]]]).cuda(), x, q, im)

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

                yhat = net.forward(torch.Tensor([name_lookup[k[0]]]).cuda(), x, q, im)
                loss = loss_fn(yhat, y)

                val_losses.append(loss.item())

                types += k
                if use_cuda:
                    y = y.cpu()
                    yhat = yhat.cpu()
                ys += y.numpy().tolist()
                yhats += yhat.detach().numpy().tolist()

            if pviz:
                viz.plot_y_yhat(ys, yhats, types, title='PolVis')

            print('[Epoch {}] - Validation Loss: {}'.format(ex, np.mean(val_losses)))
            if np.mean(val_losses) < best_val:
                best_val = np.mean(val_losses)
                file_name = 'data/models/'+model_file_name[:-3]+'_'+str(n_train)+'.pt'
                torch.save(net.state_dict(), file_name)
    return best_val


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
        train_eval(args, args.ntrain, data_file_name, args.model_fname, True, args.use_cuda)
    elif args.mode == 'ntrain':
        vals = []
        step = 500
        ns = range(step, args.ntrain, step)
        for n in ns:
            best_val = train_eval(args, n, data_file_name, args.model_fname, False, args.use_cuda)
            vals.append(best_val)
            print(n, best_val)

        import matplotlib.pyplot as plt
        plt.xlabel('n train')
        plt.ylabel('Val MSE')

        plt.plot(ns, vals)
        plt.show()
