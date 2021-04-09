import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from learning.modules.policy_encoder import PolicyEncoder
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as SpatialEncoder
from learning.modules.image_encoder import ImageEncoder as CNNEncoder
from learning.models.nn_disp_pol_vis import DistanceRegressor
from learning.dataloaders import setup_data_loaders, parse_pickle_file
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from utils.util import load_model, read_from_file
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, mech_encoding, 
                       pol_dim=3, pol_h_dim=64, pol_dropout=0., gp_pol_dim=2,
                       im_dim=3, im_h_dim=64, im_dropout=0., gp_im_dim=2):
        super(FeatureExtractor, self).__init__()
        self.mech_encoding = mech_encoding
        self.c, self.h, self.w = 3, 59, 58
        kernel_size, padding = 3, 1


        hidden_nonlin, final_nonlin = nn.ReLU, nn.ReLU
        self.pol_net = nn.Sequential(*[#nn.Linear(pol_dim, pol_h_dim), hidden_nonlin(), nn.Dropout(pol_dropout),
                                       #nn.Linear(pol_h_dim, pol_h_dim), hidden_nonlin(), nn.Dropout(pol_dropout),
                                       nn.Linear(pol_dim, gp_pol_dim), final_nonlin()])

        if mech_encoding == 'params':
            self.im_net = nn.Sequential(*[nn.Linear(im_dim, im_h_dim), hidden_nonlin(), nn.Dropout(im_dropout),
                                          nn.Linear(im_h_dim, im_h_dim), hidden_nonlin(), nn.Dropout(im_dropout),
                                          nn.Linear(im_h_dim, im_h_dim), hidden_nonlin(), nn.Dropout(im_dropout),
                                          nn.Linear(im_h_dim, gp_im_dim), final_nonlin()])
        elif mech_encoding == 'img':
            self.im_net = nn.Sequential(*[nn.Conv2d(in_channels=im_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding), 
                                          hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                          nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                          hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                          nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                          hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                          nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                          hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                          nn.Flatten(), 
                                          nn.Linear(9*im_h_dim, im_h_dim), hidden_nonlin(), nn.Dropout(im_dropout),
                                          nn.Linear(im_h_dim, gp_im_dim), final_nonlin()])
        
    def forward(self, im, mech, theta):
        if self.mech_encoding == 'img':
            # Downsample
            bs, c, h, w = im.shape
            im = im[:, :, ::2, ::2]
        else:
            im = mech
        
        #theta = self.pol_net(theta)
        im = self.im_net(im)
        
        x = torch.cat([theta, im], dim=1)
        return x

class ApproxProductDistanceGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, gp_pol_dims, gp_im_dims):
        variational_dist = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strat = VariationalStrategy(self, inducing_points, variational_dist)
        super(ApproxProductDistanceGP, self).__init__(variational_strat)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=gp_pol_dims+gp_im_dims, 
        ),#lengthscale_constraint=Interval(1.e-6, 1.0)), # 1.e-6, .25
                                        outputscale_constraint=Interval(0.01, 1.e-1)) # 0.01, 1.e-1 # 0.0225, 0.1 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ProductDistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gp_pol_dims, gp_im_dims):
        super(ProductDistanceGP, self).__init__(train_x, train_y, likelihood)        

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=gp_pol_dims+gp_im_dims), 
                                                  #lengthscale_constraint=Interval([0., 0., 0., 0., 0., 0.], [0.2, 0.01, 0.2, 1., 1., 1.])), # 1.e-6, .25
                                        outputscale_constraint=Interval(0.01, 1.e-1)) # 0.01, 1.e-1 # 0.0225, 0.1 
        #self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=gp_pol_dims+gp_im_dims))

    def forward(self, x):
        # The standardization shouldn't change at test time.
        # x = x - x.min(0)[0]
        # x = 2*(x/x.max(0)[0])-1
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def convert_training_data(train_set):
    """ GPyTorch requires the data in as two tensors: NxD, N
    """
    policies = torch.stack(train_set.dataset.tensors)
    ims = torch.stack(train_set.dataset.images)
    ys = torch.tensor(train_set.dataset.ys)
    print('Policy:', type(policies), policies.shape)
    print('Im:', type(ims), ims.shape)
    print('Y:', type(ys), ys.shape)
    xs = torch.cat([policies, 
                    ims.reshape(ims.shape[0], ims.shape[1]*ims.shape[2]*ims.shape[3])], dim=1)

    return xs, ys


def extract_feature_dataset(dataset, extractor, use_cuda=False, mech_encoding='params'):
    """ First extract the features from the dataset."""
    features, ys = [], []
    for policy_type, theta, im, y, mech in dataset:
        if use_cuda:
            theta = theta.cuda()
            im = im.cuda()
            y = y.cuda()
            mech = mech.cuda()
        feats = extractor(policy_type, im, mech, theta, mech_encoding=mech_encoding) 
        features.append(feats.detach())
        ys.append(y)
    return torch.cat(features, dim=0), torch.cat(ys, dim=0).squeeze()


def remove_duplicates(results):
    valid = []
    print('Removing duplicates...')

    found = 0
    for bb_result in results:
        valid.append([])
        added = []
        for entry in bb_result:
            policy_params = []
            for param in entry.policy_params.param_data:
                if entry.policy_params.param_data[param].varied:
                    policy_params.append(entry.policy_params.params[param])
            policy_params = np.array(policy_params)
            # Check if policy_params have already been added.
            duplicate = False
            for pol in added:
                if np.linalg.norm(pol-policy_params) < 0.05:
                    duplicate = True
            if not duplicate:# and entry.net_motion > 0.03:
                added.append(policy_params)
                valid[-1].append(entry)
            else:
                found += 1
                
        print(len(valid[-1]))
    print('Duplicates Found:', found)
    return valid


def train_exact_gp(args, train_set, val_set):
    gp_pol_dim, gp_im_dim = 3, 3

    # Extract features for this dataset and normalize them.
    print('Extracting Features')
    extractor = FeatureExtractor(mech_encoding=args.mech_type,
                                 gp_im_dim=gp_im_dim,
                                 gp_pol_dim=gp_pol_dim,
                                 pol_dropout=0.,
                                 pol_h_dim=16,
                                 im_h_dim=8,
                                 im_dropout=0.1).cuda()

    # train_x, train_y = extract_feature_dataset(train_set, extractor, use_cuda=CUDA, mech_encoding=args.mech_type)
    # mu = torch.mean(train_x, dim=0, keepdims=True)
    # std = torch.std(train_x, dim=0, keepdims=True)+1e-3
    # train_xs = ((train_x-mu)/std).detach().clone()
    # val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)
    # val_x = (val_x - mu)/std
    print('Data Size:', len(train_set.dataset))
    
    #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 6.25e-6)).cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-4)).cuda()#,
                                                         #noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(1), scale=torch.ones(1)*1e-5)).cuda()
    gp = ProductDistanceGP(train_x=torch.ones(len(train_set.dataset), gp_pol_dim+gp_im_dim).cuda(),
                           train_y=torch.ones(len(train_set.dataset)).cuda(),
                           likelihood=likelihood,
                           gp_pol_dims=gp_pol_dim,
                           gp_im_dims=gp_im_dim).cuda()
    gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3},
                                  {'params': extractor.parameters(), 'lr': 1e-3}])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    val_loss_fn = torch.nn.MSELoss()
    best = 1000

    with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
        for _, theta, im, y, mech in train_set:
            theta = theta.cuda()
            im = im.cuda()
            y = y.squeeze().cuda()
            mech = mech.cuda()
        # Normalize data.
        # print(mech.std(dim=0), theta.std(dim=0))
        # mech = (mech - mech.mean(dim=0))/mech.std(dim=0)
        # theta = (theta - theta.mean(dim=0))/theta.std(dim=0)
        
        for tx in range(50000):
            optimizer.zero_grad()            

            z = extractor(im, mech, theta)
            gp.set_train_data(inputs=z, targets=y)
            output = gp(z)

            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

            if tx % 100 == 0:
                print(tx, loss.item(), gp.likelihood.noise.item(), gp.covar_module.base_kernel.lengthscale, gp.covar_module.outputscale.item())

            if tx % 500 == 0:
                gp.eval()
                likelihood.eval()
                for _, val_theta, val_im, val_y, val_mech in val_set:
                    val_theta = val_theta.cuda()
                    val_im = val_im.cuda()
                    val_y = val_y.squeeze().cuda()
                    val_mech = val_mech.cuda()
                val_z = extractor(val_im, val_mech, val_theta)
                val_output = gp(val_z)
                train_output = gp(z)
                print(val_output.mean[0:5], val_y[0:5])
                val_loss = val_loss_fn(val_output.mean, val_y)
                train_loss = val_loss_fn(train_output.mean, y)
                print('Train Loss:', np.sqrt(train_loss.item()))
                print('Val Loss:', np.sqrt(val_loss.item()))
                gp.train()
                likelihood.train()
                
            if tx % 200 == 0:
                torch.save((extractor.state_dict(), gp.state_dict(), z, y), args.save_path)
                        
            #     print('Saving to', args.save_path)
            #     torch.save((gp.state_dict(), train_xs, train_y, mu, std), args.save_path)
            #     best = loss.item()

    gp.eval()
    print('Val Predictions')
    for ix in range(0, 20):#val_x.shape[0]):
        pred = likelihood(gp(val_x[ix:ix+1, :]))
        lower, upper = pred.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        true = val_y[ix].item()
        
        p = pred.mean.cpu().detach().numpy()[0]
        if true < lower or true > upper:
            print('INCORRECT:', pred.mean.cpu().detach().numpy(), 
                (lower[0], upper[0]), upper - lower,
                val_y[ix])
        elif upper[0]-lower[0] > 0.05:
            print('UNSURE:', pred.mean.cpu().detach().numpy(), 
                (lower[0], upper[0]), upper - lower,
                val_y[ix])
        else:
            print('CORRECT')

    print('Train Predictions')
    for ix in range(0, 20):
        pred = likelihood(gp(train_xs[ix:ix+1, :]))
        lower, upper = pred.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        print(pred.mean.cpu().detach().numpy(), 
              (lower[0], upper[0]), upper - lower,
              train_y[ix])


def train_approx_gp(args, train_set, val_set):
    gp_pol_dim, gp_im_dim = 2, 2

    # Extract features for this dataset and normalize them.
    print('Extracting Features')
    extractor = FeatureExtractor(mech_encoding=args.mech_type,
                                 gp_im_dim=gp_im_dim,
                                 gp_pol_dim=gp_pol_dim).cuda()
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 6.25e-6)).cuda()
    gp = ApproxProductDistanceGP(torch.FloatTensor(1000, gp_pol_dim+gp_im_dim).uniform_(-1, 1),
                                      gp_pol_dims=gp_pol_dim,
                                      gp_im_dims=gp_im_dim).cuda()
    gp.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3},
                                  {'params': extractor.parameters(), 'lr': 1e-3},
                                  {'params': likelihood.parameters(), 'lr': 1e-3}])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    val_loss_fn = torch.nn.MSELoss()
    best = 1000

    #with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
    for tx in range(50000):
        # This should only be one batch in exact mode.
        for _, theta, im, y, mech in train_set:
            theta = theta.cuda()
            im = im.cuda()
            y = y.squeeze().cuda()
            mech = mech.cuda()
            optimizer.zero_grad()

            z = extractor(im, mech, theta)
            output = gp(z)

            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

        if tx % 1 == 0:
            print(tx, loss.item(), likelihood.noise.item(), gp.covar_module.base_kernel.lengthscale, gp.covar_module.outputscale.item())


CUDA = True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='')
    #parser.add_argument('--pretrained-nn', type=str, required=True)
    parser.add_argument('--L', required=True, type=int)
    parser.add_argument('--mech-type', required=True, choices=['params', 'img'])
    parser.add_argument('--gp-type', required=True, choices=['exact', 'approx'])
    args = parser.parse_args()
    print(args)

    #  Load dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    results = [bb[::] for bb in raw_results[:args.L]]  
    results = remove_duplicates(results)
    results = [item for sublist in results for item in sublist[0::4]]
    data = parse_pickle_file(results)
    
    val_results = [bb[::] for bb in raw_results[81:]]
    val_results = [item for sublist in val_results for item in sublist]
    val_data = parse_pickle_file(val_results)
    val_set = setup_data_loaders(data=val_data, batch_size=len(val_data), single_set=True)

    if args.gp_type == 'exact':
        train_set = setup_data_loaders(data=data, batch_size=len(data), single_set=True)
        train_exact_gp(args, train_set, val_set)
    elif args.gp_type == 'approx':
        train_set = setup_data_loaders(data=data, batch_size=1000, single_set=True)
        train_approx_gp(args, train_set, val_set)

    
