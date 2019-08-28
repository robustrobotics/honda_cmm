import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.models.nn_disp_pol_mech import DistanceRegressor as NNPolMech
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import learning.viz as viz
from collections import namedtuple
from util import util
from util.setup_pybullet import setup_env
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from gen.generator_busybox import BusyBox, Slider
from actions import policies
from learning.test_model import test_env
import os
torch.backends.cudnn.enabled = True

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')

def train_eval(args, n_train, data_type, data_dict, hdim, batch_size, pviz, writers):

    rand_train_set, rand_val_set, rand_test_set = setup_data_loaders(data_dict['random'],
                                                batch_size=batch_size,
                                                small_train=n_train)

    act_train_set, act_val_set, act_test_set = setup_data_loaders(data_dict['active'],
                                                batch_size=batch_size,
                                                small_train=n_train)
    if data_type == 'active':
        train_set = act_train_set
    else:
        train_set = rand_train_set

    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[2, 12],
                   hdim=hdim,
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

    val_errors = OrderedDict()
    val_errors['random'] = 1000
    val_errors['active'] = 1000
    # Training loop.
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
            # to do average over all batches and add that value to writer
            #writers[data_type].add_scalar('Loss/train/'+str(n_train), loss, ex)
            loss.backward()

            optim.step()

            train_losses.append(loss.item())

        print('[Epoch {}] - Training Loss: {}'.format(ex, np.mean(train_losses)))

        if ex % args.val_freq == 0:
            for val_data_type in data_dict:
                if val_data_type == 'random':
                    val_set = rand_val_set
                else:
                    val_set = act_val_set

                val_losses = []
                net.eval()

                ys, yhats, types = [], [], []
                for bx, (k, x, q, im, y, _) in enumerate(val_set):
                    pol = torch.Tensor([util.name_lookup[k[0]]])
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

                writer_key = data_type + ',' + val_data_type
                writers[writer_key].add_scalar('Loss/val/'+str(n_train), curr_val, ex)

                #print('[Epoch {}] - Validation Loss: {}'.format(ex, curr_val))
                if curr_val < val_errors[val_data_type]:
                    val_errors[val_data_type] = curr_val

                    # save model
                    dir = 'torch_models/'+data_type+'/'+val_data_type
                    file = str(n_train)+'.pt'
                    if not os.path.isdir(dir):
                        os.makedirs(dir)
                    path = dir + '/' + file
                    torch.save(net.state_dict(), path)

                    ## save plot of prediction error
                    #if pviz:
                    #    viz.plot_y_yhat(ys, yhats, types, ex, plot_fname, title='PolVis')
    for val_data_type in data_dict:
        writer_key = data_type + ',' + val_data_type
        writers[writer_key].add_scalar('ntrain_val_loss', val_errors[val_data_type], n_train)
        
    #return val_errors, best_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--mode', choices=['ntrain', 'normal'], default='normal')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--random-data-path', type=str, required=True) # always needed for validation
    parser.add_argument('--active-data-path', type=str)
    parser.add_argument('--model-prefix', type=str, default='model')
    # in normal mode: if 0 use all samples in dataset, else use ntrain number of samples
    # in ntrain mode: must be the max number of samples you want to train with
    parser.add_argument('--n-train', type=int, default=0)
    parser.add_argument('--step', type=int, default=1000)
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

    # get list of data_paths to try
    random_data = parse_pickle_file(args.random_data_path)
    active_data = parse_pickle_file(args.active_data_path)
    data_dict = {'random': random_data,
                    'active': active_data}
    writers = {'random,random': SummaryWriter('./runs/random_random'),
                'random,active': SummaryWriter('./runs/random_active'),
                'active,active': SummaryWriter('./runs/active_active'),
                'active,random': SummaryWriter('./runs/active_random'),}

    if args.mode == 'normal':
        for data_type in data_dict:
            run_data = []
            for n in range(args.n_runs):
                for hdim in hdims:
                    for batch_size in batch_sizes:
                        plot_fname = args.model_prefix+'_nrun_'+str(n)+'_'+data_type
                        train_eval(args, 0, data_type, data_dict, hdim, batch_size, False, plot_fname, writers)
                        run_data += [RunData(hdim, batch_size, n, args.n_epochs, best_epoch, min(val_errors.keys()))]
            util.write_to_file(plot_fname+'_results', run_data)
    elif args.mode == 'ntrain':
        ns = range(args.step, args.n_train+1, args.step)
        for n_train in ns:
            for data_type in data_dict:
                train_eval(args, n_train, data_type, data_dict, args.hdim, args.batch_size, False, writers)
    for writer in writers.values():
        writer.close()
