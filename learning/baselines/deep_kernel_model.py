import argparse
import gpytorch
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import Interval, GreaterThan

from learning.baselines.datagen import setup_data_loaders


class FeatureExtractor(nn.Module):
    def __init__(self, im_dim=3, im_h_dim=8, im_dropout=0., gp_im_dim=2):
        super(FeatureExtractor, self).__init__()
        self.c, self.h, self.w = 3, 59, 58
        kernel_size, padding = 3, 1

        hidden_nonlin, final_nonlin = nn.ReLU, nn.Tanh
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
        
    def forward(self, im):
        # Downsample
        bs, c, h, w = im.shape
        im = im[:, :, ::2, ::2]
        x = self.im_net(im)
        return x


class ProductDistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gp_im_dims):
        super(ProductDistanceGP, self).__init__(train_x, train_y, likelihood)        

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=gp_im_dims), 
                                                 #lengthscale_constraint=Interval(1.e-6, 1.0)), # 1.e-6, .25
                                       outputscale_constraint=Interval(0.01, 1.e-1)) # 0.01, 1.e-1 # 0.0225, 0.1 
        #self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=gp_im_dims))

    def forward(self, x):
        # The standardization shouldn't change at test time.
        # x = x - x.min(0)[0]
        # x = 2*(x/x.max(0)[0])-1
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def evaluate_gp(gp, extractor, dataset, use_cuda):
    """
    Return the RMSE of the GP predictions on the given dataset.
    """
    val_loss_fn = torch.nn.MSELoss()
    gp.eval()
    for im, y in dataset:
        y = y.squeeze()
        if use_cuda:
            im = im.cuda()
            y = y.cuda()
        z = extractor(im)
        output = gp(z)
        val_loss = val_loss_fn(output.mean, y)
    gp.train()
    return torch.sqrt(val_loss)
    
def train_exact_gp(args, train_set, val_set, use_cuda):
    gp_im_dim = 2

    # Setup the feature extractor.
    print('Extracting Features')
    extractor = FeatureExtractor(gp_im_dim=gp_im_dim,
                                 im_dropout=0.1,
                                 im_h_dim=16)
    if use_cuda:
        extractor = extractor.cuda()
    
    # Setup the GP components.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-4)).cuda()
    #likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-7)),
    if use_cuda:
        likelihood = likelihood.cuda()
    
    init_x = torch.ones(len(train_set.dataset), gp_im_dim)
    init_y = torch.ones(len(train_set.dataset))
    if use_cuda:
        init_x = init_x.cuda()
        init_y = init_y.cuda()
    
    gp = ProductDistanceGP(train_x=init_x,
                           train_y=init_y,
                           likelihood=likelihood,
                           gp_im_dims=gp_im_dim)
    if use_cuda:
        gp = gp.cuda()

    gp.train()
    likelihood.train()

    # Setup optimizers and loss.
    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3},
                                  {'params': extractor.parameters(), 'lr': 1e-3}])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

    # Training loop.
    best = 1000
    #with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
    for im, y in train_set:
        y = y.squeeze()
        if use_cuda:
            im = im.cuda()
            y = y.cuda()

    
    for tx in range(50000):
        optimizer.zero_grad()            

        z = extractor(im)
        gp.set_train_data(inputs=z, targets=y)
        output = gp(z)

        loss = -mll(output, y)
        loss.backward()
        optimizer.step()

        if tx % 100 == 0:
            #print(list(extractor).)
            print(tx, loss.item(), gp.likelihood.noise.item(), gp.covar_module.base_kernel.lengthscale.detach(), gp.covar_module.outputscale.item())
            train_rmse = evaluate_gp(gp, extractor, train_set, use_cuda)
            val_rmse = evaluate_gp(gp, extractor, val_set, use_cuda)
            print('Train RMSE:', train_rmse)
            print('Val RMSE:', val_rmse)

                            
            # if tx % 50 == 0 and loss.item() < best:
            #     print('Saving to', args.save_path)
            #     torch.save((gp.state_dict(), train_xs, train_y, mu, std), args.save_path)
            #     best = loss.item()

    # gp.eval()
    # print('Val Predictions')
    # for ix in range(0, 20):#val_x.shape[0]):
    #     pred = likelihood(gp(val_x[ix:ix+1, :]))
    #     lower, upper = pred.confidence_region()
    #     lower = lower.cpu().detach().numpy()
    #     upper = upper.cpu().detach().numpy()
    #     true = val_y[ix].item()
        
    #     p = pred.mean.cpu().detach().numpy()[0]
    #     if true < lower or true > upper:
    #         print('INCORRECT:', pred.mean.cpu().detach().numpy(), 
    #             (lower[0], upper[0]), upper - lower,
    #             val_y[ix])
    #     elif upper[0]-lower[0] > 0.05:
    #         print('UNSURE:', pred.mean.cpu().detach().numpy(), 
    #             (lower[0], upper[0]), upper - lower,
    #             val_y[ix])
    #     else:
    #         print('CORRECT')

    # print('Train Predictions')
    # for ix in range(0, 20):
    #     pred = likelihood(gp(train_xs[ix:ix+1, :]))
    #     lower, upper = pred.confidence_region()
    #     lower = lower.cpu().detach().numpy()
    #     upper = upper.cpu().detach().numpy()
    #     print(pred.mean.cpu().detach().numpy(), 
    #           (lower[0], upper[0]), upper - lower,
    #           train_y[ix])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='')
    parser.add_argument('--L', required=True, type=int)
    args = parser.parse_args()
    print(args)

    #  Load dataset.
    with open('test.pkl', 'rb') as handle:
        raw_data = pickle.load(handle)[:args.L]
    train_set, val_set, test_set = setup_data_loaders(data=raw_data, 
                                                      batch_size=-1)
    train_exact_gp(args, train_set, val_set, use_cuda=torch.cuda.is_available())


    
