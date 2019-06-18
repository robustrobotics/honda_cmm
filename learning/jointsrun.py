import argparse
import os
import time

from learning.jointsdata import JointsDataset
from learning.jointsmodel import Statistician
from learning.jointsplot import scatter_contexts, sample_configurations
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--n-datasets', type=int, default=10000, metavar='N',
                    help='number of synthetic datasets in collection (default: 10000)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size (of datasets) for training (default: 16)')
parser.add_argument('--sample-size', type=int, default=20,
                    help='number of samples per dataset (default: 200)')
parser.add_argument('--n-features', type=int, default=3,
                    help='number of features per sample (default: 1)')
parser.add_argument('--c-dim', type=int, default=2,
                    help='dimension of c variables (default: 3)')
parser.add_argument('--n-hidden-statistic', type=int, default=5,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=128,
                    help='dimension of hidden layers in statistic network (default: 128)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=1,
                    help='dimension of z variables (default: 32)')
parser.add_argument('--n-hidden', type=int, default=5,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 3)')
parser.add_argument('--hidden-dim', type=int, default=64,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 128)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all learnable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs for training (default: 50)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
args = parser.parse_args()
assert args.output_dir is not None
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")


def run(model, optimizer, loaders, datasets):
    train_loader, test_loader = loaders
    train_dataset, test_dataset = datasets

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    alpha = 1
    tbar = tqdm(range(args.epochs))

    sample_zs = np.zeros((args.batch_size, args.sample_size, 1), dtype=np.float32)
    for ix in range(0, args.batch_size):
        sample_zs[ix, :, 0] = np.linspace(-2, 2, num=args.sample_size)
    sample_zs = torch.tensor(sample_zs).cuda().view(-1, 1)
    print(sample_zs.shape)

    # main training loop
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        running_kl = 0
        running_recon = 0
        for batch in train_loader:
            vlb, recon, kl = model.step(batch, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb
            running_kl += kl
            running_recon += recon

        running_vlb /= (len(train_dataset) // args.batch_size)
        running_recon /= (len(train_dataset) // args.batch_size)
        running_kl /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f} RECON: {:.3f} KL: {:.3f}".format(running_vlb, running_recon, running_kl)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # show test set in context space at intervals
        if (epoch + 1) % 1 == 0:
            model.eval()
            contexts = []
            true_xs = []
            sampled_xs = []
            recon_xs = []
            true_xs2 = []
            sampled_xs2 = []
            recon_xs2 = []
            for batch in test_loader:
                if len(contexts) > 50: break
                inputs = Variable(batch.cuda(), volatile=True)
                context_means, _ = model.statistic_network(inputs)
                contexts.append(context_means.data.cpu().numpy())

                hidden, hidden_var = model.inference_networks[0](inputs, None, context_means)
                recon, recon_logvar = model.observation_decoder(hidden, context_means)
                sampled_configurations, _ = model.observation_decoder(sample_zs, context_means)
                sampled_configurations = sampled_configurations.view(args.batch_size, args.sample_size, 3)
                recon = recon.view(args.batch_size, args.sample_size, 3)

                true_xs.append(batch[0, :, :].data.cpu().numpy())
                sampled_xs.append(sampled_configurations[0, :, :].data.cpu().numpy())
                recon_xs.append(recon[0, :, :].data.cpu().numpy())

                true_xs2.append(batch[1, :, :].data.cpu().numpy())
                sampled_xs2.append(sampled_configurations[1, :, :].data.cpu().numpy())
                recon_xs2.append(recon[1, :, :].data.cpu().numpy())

            #print(recon_logvar[0,:])#, hidden_var[0,:])
            # show coloured by joint type
            # path = args.output_dir + '/figures/' + time_stamp + '-{}.pdf'.format(epoch + 1)
            # scatter_contexts(contexts, test_dataset.data['labels'],
            #                  test_dataset.data['joints'], savepath=path)

            # Plot configurations by sampling from the latent space.
            path = args.output_dir + '/figures/configurations-{0}'.format(epoch+1)
            sample_configurations(true_xs, sampled_xs, recon_xs, path)

            path = args.output_dir + '/figures/configurations-{0}'.format(epoch + 1)
            sample_configurations(true_xs2, sampled_xs2, recon_xs2, path, suffix='-2')
        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            save_path = args.output_dir + '/checkpoints/' + time_stamp \
                        + '-{}.m'.format(epoch + 1)
            model.save(optimizer, save_path)


def main():
    train_dataset = JointsDataset(fname='ns_large/dataset_test.pkl')

    n_test_datasets = args.n_datasets // 10
    test_dataset = JointsDataset(fname='ns_large/dataset_test.pkl')

    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=True)
    loaders = (train_loader, test_loader)

    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': args.n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.relu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run(model, optimizer, loaders, datasets)


if __name__ == '__main__':
    main()
