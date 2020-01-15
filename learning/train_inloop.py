from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import argparse
from argparse import Namespace
from gen.generate_policy_data import generate_dataset
from learning.gp.explore_single_bb import create_single_bb_gpucb_dataset
from learning.train import train_eval
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


def create_dataset_while_training_nn(n_interactions, n_bbs, args):
    """
    :param n_bbs: The number of BusyBoxes to include in the dataset.
    :param n_interactions: The number of interactions per BusyBox.
    :param args:
    :return:
    """
    # Create a dataset of L busyboxes.
    if args.bb_fname == '':
        bb_dataset_args = Namespace(max_mech=1,
                                    mech_types=args.mech_types,
                                    urdf_num=args.urdf_num,
                                    debug=False,
                                    n_bbs=n_bbs,
                                    n_samples=1,
                                    viz=False,
                                    random_policies=False,
                                    randomness=1.0,
                                    goal_config=None,
                                    bb_fname=None,
                                    no_gripper=True)
        busybox_data = generate_dataset(bb_dataset_args, None)
        print('BusyBoxes created.')
    else:
        # Load in a file with predetermined BusyBoxes.
        with open(args.bb_fname, 'rb') as handle:
            busybox_data = pickle.load(handle)
    busybox_data = [bb_results[0] for bb_results in busybox_data][:n_bbs]

    # Create folders to save data and models.
    model_dir = './' + args.save_dir + '/torch_models'
    runs_dir = './' + args.save_dir + '/runs'
    data_dir = './' + args.save_dir + '/data'
    os.makedirs('./' + args.save_dir + '/')
    os.makedirs(model_dir)
    os.makedirs(runs_dir)
    os.makedirs(data_dir)
    writer = SummaryWriter(runs_dir)

    dataset = []
    results = []
    regrets = []
    nn_fname = ''
    for ix, bb_results in enumerate(busybox_data):
        # Sample a dataset with the most recent NN.
        single_dataset, _, r = create_single_bb_gpucb_dataset(bb_results,
                                                              n_interactions,
                                                              nn_fname,
                                                              args.plot,
                                                              args,
                                                              ix,
                                                              ret_regret=True)
        dataset.append(single_dataset)
        results.extend(single_dataset)

        if (ix+1) % args.train_freq == 0:
            print('Training with %d busyboxes.' % (ix+1))
            # Save the current dataset.
            with open('{}/data_{}L_{}M.pickle'.format(data_dir, ix+1, args.M), 'wb') as handle:
                pickle.dump(dataset, handle)
            # Train the NN.
            train_eval(args=args,
                       hdim=args.hdim,
                       batch_size=args.batch_size,
                       pviz=False,
                       results=results,
                       fname='{}/model_{}L_{}M'.format(model_dir, ix+1, args.M),
                       writer=writer)
            nn_fname = '{}/model_{}L_{}M.pt'.format(model_dir, ix+1, args.M)

        regrets.append(r)
        print('Interacted with BusyBox %d.' % ix)
        print(r)
    print('Regret:', np.mean(regrets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-gp-samples',
        type=int,
        default=500,
        help='number of samples to use when fitting a GP to data')
    parser.add_argument(
        '--M',
        type=int,
        help='number of interactions within a single BusyBox during training time')
    parser.add_argument(
        '--L',
        type=int,
        help='number of BusyBoxes to interact with during training time')
    parser.add_argument(
        '--urdf-num',
        default=0,
        help='number to append to generated urdf files. Use if generating multiple datasets simultaneously.')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--mech-types',
        nargs='+',
        default='slider',
        type=str,
        help='if no bb-fname is specified, list the mech types desired')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='use to generate polar plots during GP-UCB interactions')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='use to sample from the acquistion function instead of optimizing')
    parser.add_argument(
        '--train-freq',
        type=int,
        default=5,
        help='After how many BusyBoxes we should train the NN.')

    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--use-cuda', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--save-dir', required=True, type=str)
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])

    args = parser.parse_args()
    print(args)

    if args.debug:
        import pdb
        pdb.set_trace()

    create_dataset_while_training_nn(args.M, args.L, args)
