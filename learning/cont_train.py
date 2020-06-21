import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders, parse_pickle_file
from learning.gp.explore_single_bb import create_gpucb_dataset
import learning.viz as viz
from collections import namedtuple
from utils import util
import os
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from actions.policies import Policy
from argparse import Namespace
torch.backends.cudnn.enabled = True

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')
name_lookup = {'Prismatic': 0, 'Revolute': 1}


def view_points(img, points):
    c, h, w = img.shape
    img = img / 2 + 0.5
    npimg = img.numpy()

    fig, axes = plt.subplots(1, 1)
    axes.imshow(np.transpose(npimg, (1, 2, 0)))
    cmap = plt.get_cmap('viridis')

    for ix in range(0, points.sxhape[0]):
        axes.scatter((points[ix, 0]+1)/2.0*w, (points[ix, 1]+1)/2.0*h,
                     s=5, c=[cmap(ix/points.shape[0])])

    return fig


def train_eval(args, hdim, batch_size, pviz, fname, writer):

    # Set up dataloaders for test set
    test_results = util.read_from_file('test20.pickle')
    for num in range(args.L_min, args.L_max + 1, args.L_step):
        new_results = []
        for res in test_results[0:num]:
            new_results += res[0:args.M]
    test_data = parse_pickle_file(new_results)
    test_set = setup_data_loaders(data=test_data, batch_size=batch_size, single_set=True)

    # Setup Model
    policy_types = ['Prismatic', 'Revolute']
    net = NNPolVis(policy_names=policy_types,
                   policy_dims=Policy.get_param_dims(policy_types),
                   hdim=hdim,
                   im_h=53,  # 154, Note these aren't important for the SpatialAutoencoder
                   im_w=115,  # 205,
                   image_encoder=args.image_encoder)

    if args.use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(net.parameters())

    # Get initial 500 samples (5 busyboxes with 100 interactions each)
    train_args = Namespace(n_gp_samples=500, bb_fname='', mech_types=['slider'], plot=False, urdf_num=0,
                           fname='', nn_fname='', plot_dir='', debug=False, random_policies=False, stochastic=False)
    dataset = create_gpucb_dataset(100, 5, train_args, net)
    net.train()
    data = []
    # Append samples from the dataset to data
    for i in range(len(dataset)):
        data.extend(parse_pickle_file(dataset[i]))
    buffer = []  # Replay buffer
    new_samples = []
    count = 0  # Count number of samples seen so far

    for k in range(20):   # CHANGE to a variable
        # Get 500 new samples (5 busyboxes with 100 interactions each)
        train_args = Namespace(n_gp_samples=500, bb_fname='', mech_types=['slider'], plot=False, urdf_num=0,
                               fname='', nn_fname='', plot_dir='', debug=False, random_policies=False, stochastic=False)
        new_dataset = create_gpucb_dataset(100, 5, train_args, net)
        net.train()
        for j in range(len(new_dataset)):
            data.extend(parse_pickle_file(new_dataset[j]))
        print('data length: ' + str(len(data)))
        while count < len(data):
            # Cap buffer size at 1000
            new_samples.append(data[count])
            count += 1
            # Load 50 new samples into the buffer at a time
            if len(new_samples) == 50:
                while len(buffer) > 950:
                    buffer.pop(random.randint(0, len(buffer) - 1))
                # Include whole buffer when training
                buffer.extend(new_samples)
                train_set, val_set, _ = setup_data_loaders(data=buffer, batch_size=batch_size)
                new_samples = []

                # Training loop.
                for ex in range(1, args.n_epochs+1):
                    print('training...')
                    net.train()
                    for bx, (k, x, im, y, _) in enumerate(train_set):
                        pol = name_lookup[k[0]]
                        if args.use_cuda:
                            x = x.cuda()
                            im = im.cuda()
                            y = y.cuda()
                        optim.zero_grad()
                        yhat, points = net.forward(pol, x, im)

                        loss = loss_fn(yhat, y)
                        loss.backward()

                        optim.step()

                    # Calculate training loss after each busybox is added (average on all previously seen samples)
                    if count % 100 == 0 and ex == args.n_epochs:
                        train_losses = []
                        seen_samples = data[:i]
                        sample_set = setup_data_loaders(data=seen_samples, batch_size=batch_size, single_set=True)
                        ys, yhats, types = [], [], []
                        for bx, (k, x, im, y, _) in enumerate(sample_set):
                            pol = torch.Tensor([name_lookup[k[0]]])
                            if args.use_cuda:
                                x = x.cuda()
                                im = im.cuda()
                                y = y.cuda()

                            yhat, _ = net.forward(pol, x, im)

                            loss = loss_fn(yhat, y)
                            train_losses.append(loss.item())

                            types += k
                            if args.use_cuda:
                                y = y.cpu()
                                yhat = yhat.cpu()
                            ys += y.numpy().tolist()
                            yhats += yhat.detach().numpy().tolist()

                        curr_val = np.mean(train_losses)
                        writer.add_scalar('Train-loss/'+fname, curr_val, ex)
                        print('[Busybox {}] - Training Loss: {}'.format(count/100, curr_val))

                    # Calculate validation error on held out test set
                        val_losses = []
                        net.eval()
                        ys, yhats, types = [], [], []
                        for bx, (k, x, im, y, _) in enumerate(test_set):
                            pol = torch.Tensor([name_lookup[k[0]]])
                            if args.use_cuda:
                                x = x.cuda()
                                im = im.cuda()
                                y = y.cuda()

                            yhat, _ = net.forward(pol, x, im)

                            loss = loss_fn(yhat, y)
                            val_losses.append(loss.item())

                            types += k
                            if args.use_cuda:
                                y = y.cpu()
                                yhat = yhat.cpu()
                            ys += y.numpy().tolist()
                            yhats += yhat.detach().numpy().tolist()

                        curr_val_error = np.mean(val_losses)
                        writer.add_scalar('Val-loss/'+fname, curr_val_error, ex)
                        print('[Busybox {}] - Validation Loss: {}'.format(count/100, curr_val_error))

                        # save model for every 5 busyboxes
                        if count % 500 == 0:
                            full_path = fname+str(count)+'.pt'
                            torch.save(net.state_dict(), full_path)

                            # save plot of prediction error
                            # if pviz:
                            #     viz.plot_y_yhat(ys, yhats, types, ex, fname, title='PolVis')

def get_train_params(args):
    return {'batch_size': args.batch_size,
            'hdim': args.hdim,
            'n_epochs': args.n_epochs,
            'val_freq': args.val_freq,
            'L_min': args.L_min,
            'L_max': args.L_max,
            'L_step': args.L_step,
            'M': args.M}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--use-cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-dir', required=True, type=str)
    # if want to just train one model, make L_min == L_max and L_step == 1
    parser.add_argument('--L-min', type=int, default=10)
    parser.add_argument('--L-max', type=int, default=100)
    parser.add_argument('--L-step', type=int, default=10)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
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
    model_fname = model_dir+'model_'+str(args.M)+'M'
    train_eval(args, args.hdim, args.batch_size, args.pviz, model_fname, writer)
    writer.close()

    # save run params to text file in models dir
    import git
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch
    git_hash = repo.head.object.hexsha
    all_params = {'branch': branch, 'hash': git_hash}
    all_params.update(get_train_params(args))
    util.write_to_file(args.save_dir+'/run_params.txt', str(all_params)+'\n')
