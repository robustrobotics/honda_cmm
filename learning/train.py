import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import learning.viz as viz
from collections import namedtuple
from util import util
import os
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')
name_lookup = {'Prismatic': 0, 'Revolute': 1}


def train_eval(args, hdim, batch_size, pviz, fname, writer, n=0, data_fname=None):
    # Load data
    if args.data_fname:
        data = parse_pickle_file(fname=args.data_fname)
    else:
        data = parse_pickle_file(fname=data_fname)

    train_set, val_set, test_set = setup_data_loaders(data=data,
                                                      batch_size=batch_size,
                                                      small_train=n)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[2, 12],
                   hdim=hdim,
                   im_h=53,  # 154,
                   im_w=115,  # 205,
                   kernel_size=3,
                   image_encoder=args.image_encoder)

    #print(sum(p.numel() for p in net.parameters() if p.requires_grad))
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
        train_loss_ex = np.mean(train_losses)
        writer.add_scalar('Train-loss/'+fname, train_loss_ex, ex)
        print('[Epoch {}] - Training Loss: {}'.format(ex, train_loss_ex))

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
            writer.add_scalar('Val-loss/'+fname, curr_val, ex)
            print('[Epoch {}] - Validation Loss: {}'.format(ex, curr_val))
            # if best epoch so far, save model
            if curr_val < best_val:
                best_val = curr_val
                full_path = fname+'.pt'
                torch.save(net.state_dict(), full_path)

                # save plot of prediction error
                if pviz:
                    viz.plot_y_yhat(ys, yhats, types, ex, fname, title='PolVis')

def get_train_params(args):
    return {'batch_size': args.batch_size,
            'hdim': args.hdim,
            'n_epochs': args.n_epochs,
            'val_freq': args.val_freq,
            'data-fname': args.data_fname}

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
    parser.add_argument('--save-dir', required=True, type=str)
    # if 0 then use all samples in dataset, else use ntrain number of samples
    parser.add_argument('--ntrain-max', type=int, default=0)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    parser.add_argument('--n-runs', type=int, default=1)
    parser.add_argument('--pviz', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # remake dirs (don't want to overwrite data)
    model_dir = './'+args.save_dir+'/torch_models/'
    runs_dir = './'+args.save_dir+'/runs'
    os.makedirs('./'+args.save_dir+'/')
    os.makedirs(model_dir)
    os.makedirs(runs_dir)

    # make tensorboard writer
    writer = SummaryWriter(runs_dir)

    if args.mode == 'normal':
        for n_run in range(args.n_runs):
            fname = model_dir+'model_nrun_'+str(n_run)
            train_eval(args, args.hdim, args.batch_size, args.pviz, fname, writer)

    elif args.mode == 'ntrain':
        vals = []
        ns = range(args.step, args.ntrain_max+1, args.step)
        for n in ns:
            fname = model_dir+'model_ntrain_'+str(n)
            train_eval(args, args.hdim, args.batch_size, args.pviz, fname, writer, n)

    writer.close()

    # save run params to text file in models dir
    import git
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    git_hash = repo.head.object.hexsha
    all_params = {'branch': branch, 'hash': git_hash}
    all_params.update(get_train_params(args))
    util.write_to_file(args.save_dir+'/run_params.txt', str(all_params)+'\n')
