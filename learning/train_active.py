import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import learning.viz as viz
from collections import namedtuple
from util import util
from torch.utils.tensorboard import SummaryWriter
from gen import active_prior
from gen.generator_busybox import BusyBox, Slider
import os
torch.backends.cudnn.enabled = True

def train_eval(args, parsed_data, plot_fname, pviz=False):

    train_set = setup_data_loaders(parsed_data, batch_size=args.batch_size, train_only=True)

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[2, 12],
                   hdim=args.hdim,
                   im_h=53,  # 154,
                   im_w=115,  # 205,
                   kernel_size=3,
                   image_encoder=args.image_encoder)

    if args.use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    # Add the graph to TensorBoard viz,
    k, x, q, im, y, _ = train_set.dataset[0]
    pol = torch.Tensor([util.name_lookup[k]])
    if args.use_cuda:
        x = x.cuda().unsqueeze(0)
        q = q.cuda().unsqueeze(0)
        im = im.cuda().unsqueeze(0)
        pol = pol.cuda()

    # Training loop.
    best_train_error = 10000
    for ex in range(1, args.n_epochs+1):
        train_losses = []
        net.train()
        for bx, (k, x, q, im, y, _) in enumerate(train_set):
            pol = util.name_lookup[k[0]]
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
        best_train_error = min(best_train_error, train_loss_ex)
        print('[Epoch {}] - Training Loss: {}'.format(ex, train_loss_ex))
    return best_train_error, net.state_dict()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=16, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--plot-freq', type=int, default=5)
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    parser.add_argument('--n-max', type=int, default=3000)
    parser.add_argument('--data-type', default='active', choices=['active', 'random'])
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # get list of data_paths to try
    dir = 'torch_models_prior/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    model_path = dir + args.data_type + '.pt'
    n_samples = 200 # num samples per bb
    n_bbs = int(args.n_max/n_samples)
    dataset = []
    writer = SummaryWriter('./runs_active')
    for i in range(1, n_bbs+1):
        print('BusyBox: ', i, '/', n_bbs)
        rand_num = np.random.uniform()
        bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider], urdf_tag=str(rand_num), debug=False)
        data, prior_fig = active_prior.generate_dataset(1, n_samples, False, False, str(rand_num), 1, False, 'random' == args.data_type, bb=bb, model_path=model_path, hdim=args.hdim)
        dataset += data
        parsed_data = parse_pickle_file(data=dataset)
        plot_fname = args.data_type+'_'+str(i)
        train_error, model = train_eval(args, parsed_data, plot_fname)
        writer.add_scalar('Loss/train', train_error, i)
        writer.add_figure('Prior/'+str(i), prior_fig)
        path = dir + args.data_type + '.pt'
        torch.save(model, path)
        if i % args.plot_freq == 0:
            save_model_path = dir + plot_fname + '.pt'
            torch.save(model, path)
            # TODO: visualize predicted competence
            # writer figure to tensorboard at this frequency

    writer.close()
