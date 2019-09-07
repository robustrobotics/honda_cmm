import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import learning.viz as viz
from learning.test_model import test_env
from collections import namedtuple
from util import util
from torch.utils.tensorboard import SummaryWriter
from gen import active_prior
from gen.generator_busybox import BusyBox, Slider
import os
import shutil
import matplotlib.pyplot as plt
import traceback, sys, pdb
torch.backends.cudnn.enabled = True

def train_eval(args, parsed_data, pviz=False):

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

def get_train_params(args):
    return {'batch_size': args.batch_size,
            'hdim': args.hdim,
            'n_epochs': args.n_epochs,
            'n_bbs': args.n_bbs,
            'data_type': args.data_type,
            'n_inter': args.n_inter,
            'n_prior': args.n_prior,
            'train_freq': args.train_freq,
            'bb_file': args.bb_file}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, default=16, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    #parser.add_argument('--plot-freq', type=int, default=5)
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    parser.add_argument('--n-bbs', type=int, default=5) # maximum number of robot interactions
    parser.add_argument('--data-type', choices=['sagg', 'sagg-learner', 'random', 'random-learner'])
    parser.add_argument('--n-inter', default=20, type=int) # number of samples used during interactions
    parser.add_argument('--n-prior', default=10, type=int) # number of samples used to generate prior
    parser.add_argument('--viz-cont', action='store_true') # visualize interactions and prior as they're generated
    parser.add_argument('--viz-final', action='store_true') # visualize final interactions and priors
    parser.add_argument('--lite', action='store_true') # if used, does not generate or save any plots
    parser.add_argument('--train-freq', default=1, type=int) # frequency to retrain and test model
    parser.add_argument('--bb-file', type=str)
    parser.add_argument('--urdf-tag', type=str, default='0')
    args = parser.parse_args()

    if args.debug:
        pdb.set_trace()

    try:
        # remove directories for tensorboard logs and torch model then remake
        model_dir = './torch_models_prior/'
        runs_dir = './runs_active'
        dirs = [model_dir, runs_dir]
        for dir in dirs:
            if os.path.isdir(dir):
                input('move this directory so files dont get overitten: '+dir)
                sys.exit()
            os.makedirs(dir)
        plt.ion()

        if args.bb_file:
            bbps = util.read_from_file(args.bb_file)

        dataset = []
        writer = SummaryWriter(runs_dir)
        test_norm_regrets = []
        model_path = None
        for i in range(1,args.n_bbs+1):
            print('BusyBox: ', i, '/', args.n_bbs)
            if args.bb_file:
                bbp = bbps[i-1]
                bb = BusyBox.get_busybox(bbp.width, bbp.height, bbp._mechanisms, urdf_tag=args.urdf_tag)
            else:
                bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider], debug=False, urdf_tag=args.urdf_tag)

            # test model on this novel busybox
            if model_path:
                max_motion = bb._mechanisms[0].range/2
                model = util.load_model(model_path, hdim=args.hdim)
                pred_motion = test_env(model, bb=bb, plot=False, viz=False, debug=False, use_cuda=False)
                test_regret = (max_motion - max(0., pred_motion))/max_motion
                test_norm_regrets += [test_regret]
                writer.add_scalar('Test_Regret/Regret', test_regret, i)
                writer.add_scalar('Test_Regret/Average_Regret', np.mean(test_norm_regrets), i)

            # generate data
            if args.data_type == 'sagg':
                n_prior = 0
                random = False
            if args.data_type == 'sagg-learner':
                n_prior = args.n_prior
                random = False
            if args.data_type =='random':
                n_prior = 0
                random = True
            if args.data_type == 'random-learner':
                n_prior = args.n_prior
                random = True
            return_tup = active_prior.generate_dataset(1, \
                                                    args.n_inter,
                                                    n_prior,
                                                    False,
                                                    False,
                                                    args.urdf_tag,
                                                    1,
                                                    args.viz_final,
                                                    args.viz_cont,
                                                    random,
                                                    bb=bb, model_path=model_path,
                                                    hdim=args.hdim,
                                                    lite=args.lite)
            if args.lite:
                learner = return_tup
            else:
                learner, prior_figs, final_figs, interest_figs = return_tup
                if prior_figs[0]:
                    writer.add_figure('Prior/'+str(i), prior_figs[0])
                if final_figs[0]:
                    writer.add_figure('Final/'+str(i), final_figs[0])
                if interest_figs[0]:
                    writer.add_figure('Interest/'+str(i), interest_figs[0])
            dataset += learner.interactions

            if not i % args.train_freq:
                # train model
                parsed_data = parse_pickle_file(data=dataset)
                train_error, model = train_eval(args, parsed_data)

                # save model
                model_path = model_dir + args.data_type + str(i) + '.pt'
                torch.save(model, model_path)
                writer.add_scalar('Loss/train', train_error, i)

        writer.close()
        import git
        repo = git.Repo(search_parent_directories=True)
        branch = repo.active_branch
        git_hash = repo.head.object.hexsha
        all_params = {'branch': branch, 'hash': git_hash}
        all_params.update(learner.get_params())
        all_params.update(get_train_params(args))
        util.write_to_file('run_params.txt', str(all_params)+'\n')
    except:
        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
