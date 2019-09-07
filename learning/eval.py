import argparse
import numpy as np
import torch
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.models.nn_disp_pol_mech import DistanceRegressor as NNPolMech
from learning.dataloaders import setup_data_loaders
import learning.viz as viz
from collections import namedtuple
from util import util
torch.backends.cudnn.enabled = True

RunData = namedtuple('RunData', 'hdim batch_size run_num max_epoch best_epoch best_val_error')
name_lookup = {'Prismatic': 0, 'Revolute': 1}

def evaluate(args, hdim, batch_size, pviz, fname):
    # Load data
    train_set, _, _ = setup_data_loaders(fname=args.data_fname,
                                         batch_size=batch_size,
                                         small_train=args.n_train)
    # TODO: Load the model.
    # Setup Model (TODO: Update the correct policy dims)
    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[2, 12],
                   hdim=hdim,
                   im_h=53,  # 154,
                   im_w=115,  # 205,
                   kernel_size=3,
                   image_encoder=args.image_encoder,
                   pretrain_encoder=args.pretrain_encoder,
                   n_features=args.n_features)
    net.load_state_dict(torch.load(fname))

    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    if args.use_cuda:
        net = net.cuda()

    loss_fn = torch.nn.MSELoss()

    vals = []
    val_losses = []
    net.eval()

    for bx, (k, x, q, im, y, _) in enumerate(train_set):
        pol = torch.Tensor([name_lookup[k[0]]])
        if args.use_cuda:
            x = x.cuda()
            q = q.cuda()
            im = im.cuda()
            y = y.cuda()

        yhat = net.forward(pol, x, q, im)

        loss = loss_fn(yhat, y)
        val_losses.append(loss.item())

    curr_val = np.mean(val_losses)
    print('Validation Error:', curr_val)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size to use for training.')
    parser.add_argument('--hdim', type=int, help='Hidden dimensions for the neural nets.')
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--val-freq', type=int, default=5)
    parser.add_argument('--mode', choices=['ntrain', 'normal'], default='normal')
    parser.add_argument('--use-cuda', default=False, action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--data-fname', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='model')
    # if 0 then use all samples in dataset, else use ntrain number of samples
    parser.add_argument('--n-train', type=int, default=0)
    parser.add_argument('--image-encoder', type=str, default='spatial', choices=['spatial', 'cnn'])
    parser.add_argument('--n-runs', type=int, default=1)
    parser.add_argument('--pretrain-encoder', default='', type=str)
    parser.add_argument('--n-features', type=int, default=16)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # if hdim and batch_size given as args then use them, otherwise test a list of them
    hdim = args.hdim
    batch_size = args.batch_size

    fname = args.model_name
    evaluate(args, hdim, batch_size, False, fname)
