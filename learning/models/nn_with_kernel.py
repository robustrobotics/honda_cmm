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
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import Interval
from utils.util import load_model, read_from_file
import numpy as np

class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, pretrained_nn_path):
        super(PretrainedFeatureExtractor, self).__init__()
        self.pretrained_model = load_model(model_fname=pretrained_nn_path,
                                           hdim=16)
        
    def forward(self, policy_type, im, theta):
        policy_type = 1
        if policy_type == 0:
            policy_type = 'Prismatic'
        else:
            policy_type = 'Revolute'
        pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        im, points = self.pretrained_model.image_module(im)
        x = torch.cat([im, pol], dim=1)
        x = F.relu(self.pretrained_model.fc1(x))
        x = F.relu(self.pretrained_model.fc2(x))
        return x

    def forward_cached(self, policy_type, im, theta):
        policy_type = 'Revolute'
        pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        x = torch.cat([im, pol], dim=1)
        x = F.relu(self.pretrained_model.fc1(x))
        x = F.relu(self.pretrained_model.fc2(x))
        return x


class PretrainedDistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(PretrainedDistanceGP, self).__init__(train_x, train_y, likelihood)
        num_gp_dims = 3
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=num_gp_dims, lengthscale_constraint=gpytorch.constraints.Interval(0.0001, 1.)), outputscale_constraint=gpytorch.constraints.Interval(1e-8, 1.e-6))
        pol_dim, im_dim, h_dim = 8, 32,32
        
        self.lin = nn.Linear(h_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, num_gp_dims)
        
    
    def embed_input(self, x): 
        x = torch.tanh(self.lin(x))
        x = self.lin2(x)
        return x

    def forward(self, x):
        x = self.embed_input(x)

        # The standardization shouldn't change at test time.
        self.xmin = x.min(0)[0]
        x = x - self.xmin
        self.xmax = x.max(0)[0]
        x = 2*(x/self.xmax)-1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_nn_path):
        super(FeatureExtractor, self).__init__()
        self.pretrained_model = load_model(model_fname=pretrained_nn_path,
                                           hdim=16)
        
    def forward(self, policy_type, im, mech, theta, mech_encoding='params'):
        policy_type = 1
        if policy_type == 0:
            policy_type = 'Prismatic'
        else:
            policy_type = 'Revolute'
        
        #pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        # _, points = self.pretrained_model.image_module(im)

        if mech_encoding == 'img':
            bs, c, h, w = im.shape
            # Downsample
            im = im[:, :, ::2, ::2]
            im = im.reshape(bs, -1)
        else:
            im = mech
        
        x = torch.cat([im, theta], dim=1)
        #x = F.relu(self.pretrained_model.fc1(x))
        #x = F.relu(self.pretrained_model.fc2(x))
        return x

    def forward_cached(self, policy_type, im, theta):
        policy_type = 'Revolute'
        pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        x = torch.cat([im, theta], dim=1)
        # x = F.relu(self.pretrained_model.fc1(x))
        #x = F.relu(self.pretrained_model.fc2(x))
        return x

class PolPredictor(nn.Module):
    def __init__(self, gp):
        super(PolPredictor, self).__init__()
        self.net = gp.pol_net
        self.lin = nn.Linear(gp.gp_action_dims, 1)

    def forward(self, x):
        x = self.net(x)
        return self.lin(x)

class ImPredictor(nn.Module):
    def __init__(self, gp):
        super(ImPredictor, self).__init__()
        self.net = gp.im_net
        self.lin = nn.Linear(gp.gp_im_dims, 1)

    def forward(self, x):
        x = self.net(x)
        return self.lin(x)
        

class ProductDistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mech_encoding='params'):
        super(ProductDistanceGP, self).__init__(train_x, train_y, likelihood)
        self.gp_action_dims, self.gp_im_dims = 3, 3
        

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.gp_action_dims+self.gp_im_dims, 
        ),#lengthscale_constraint=Interval(1.e-6, 1.0)), # 1.e-6, .25
                                        outputscale_constraint=Interval(0.01, 1.e-1)) # 0.01, 1.e-1 # 0.0225, 0.1 
        pol_dim, im_dim, pol_h_dim, im_h_dim = 3, 3, 64, 64 # 128

        pol_dropout = 0.
        im_dropout = 0.

        self.pol_net = nn.Sequential(*[nn.Linear(pol_dim, pol_h_dim),
                                      nn.ReLU(),
                                      nn.Dropout(pol_dropout),
                                      nn.Linear(pol_h_dim, pol_h_dim),
                                      nn.ReLU(),
                                      nn.Dropout(pol_dropout),
                                      nn.Linear(pol_h_dim, self.gp_action_dims),
                                      nn.Tanh()])
        if mech_encoding == 'params':
            self.im_net = nn.Sequential(*[nn.Linear(im_dim, im_h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(im_dropout),
                                        nn.Linear(im_h_dim, im_h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(im_dropout),
                                        nn.Linear(im_h_dim, im_h_dim),
                                        nn.ReLU(),
                                        nn.Dropout(im_dropout),
                                        nn.Linear(im_h_dim, self.gp_im_dims)])
        elif mech_encoding == 'img':
            self.c, self.h, self.w = 3, 59, 58
            kernel_size, padding = 3, 1
            self.conv1 = nn.Conv2d(in_channels=im_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.maxpool = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
            self.conv4 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)

            self.flatten = nn.Flatten()
            self.lin1 = nn.Linear(9*im_h_dim, im_h_dim)
            self.lin2 = nn.Linear(im_h_dim, self.gp_im_dims)
            self.dropout = nn.Dropout(0.)

            self.im_net = self.im_net2

    
        # Seems to fix an issue where the embeddings of two points get very close causing numerical issues.
        # Could look into this noise issue a bit more later.
        
    def im_net2(self, im):
        im = im.view(-1, self.c, self.h, self.w)
        im = self.conv1(im)
        im = self.dropout(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.conv2(im)
        im = self.dropout(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.conv3(im)
        im = self.dropout(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.conv4(im)
        im = self.dropout(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.flatten(im)
        im = self.lin1(im)
        im = self.dropout(im)
        im = self.relu(im)
        return self.tanh(self.lin2(im))

    def embed_input(self, x): 
        theta, im = x[:, -3:], x[:,:-3]

        # First embed the policy.
        pol = self.pol_net(theta)
        im = self.im_net(im)
        
        x = torch.cat([pol, im], dim=1)
        return x


    def forward(self, x):
        x = self.embed_input(x)

        # The standardization shouldn't change at test time.
        #x = x - x.min(0)[0]
        #x = 2*(x/x.max(0)[0])-1
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DistanceGP, self).__init__(train_x, train_y, likelihood)
        num_gp_dims = 4
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=num_gp_dims, 
                                                  lengthscale_constraint=gpytorch.constraints.Interval(1.e-6, 1.)), 
                                        outputscale_constraint=gpytorch.constraints.Interval(0.0225, 1.e-1))
        pol_dim, im_dim, h_dim = 32, 32, 128
        self.pol_embed = nn.Linear(3, pol_dim)
        self.im_embed = nn.Linear(im_dim, pol_dim)
        self.lin = nn.Linear(2*pol_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, num_gp_dims)
        self.dropout = nn.Dropout(0.2) 
        #self.cuda()
        #self.init_minmax(train_x)
    
    def init_minmax(self, train_x):
        x = self.embed_input(train_x)
        self.xmin = x.min(0)[0]
        x = x - self.xmin
        self.xmax = x.max(0)[0]

    def embed_input(self, x): 
        theta, im = x[:, -3:], x[:,:-3]

        # First embed the policy and image.
        pol = self.pol_embed(theta)
        im = self.im_embed(im)
        
        embed = torch.tanh(torch.cat([pol, im], dim=1))
        x = self.dropout(embed)
        # Use skip connections to encourage the kernel to pay attention to the policy.
        x = torch.tanh(self.lin(embed))
        x = self.dropout(x)
        #x = torch.cat([x, embed], dim=1)
        x = torch.tanh(self.lin2(x))
        x = self.dropout(x)
        #x = torch.cat([x, embed], dim=1)
        x = self.lin3(x)
        return x


    def forward(self, x):
        x = self.embed_input(x)

        # The standardization shouldn't change at test time.
        x = x - x.min(0)[0]
        x = 2*(x/x.max(0)[0])-1
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        #import pickle
        #with open('debug.pickle', 'wb') as handle:
        #    pickle.dump([mean_x.clone().cpu().detach().numpy(), 
        #                 covar_x.clone().cpu().detach().numpy()], handle)
        #print(mean_x.shape, covar_x.shape)
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

def pretrain_modules(gp, train_x, train_y, epochs=5):
    print('Pretraining...')
    pol_net = PolPredictor(gp)
    pol_net = pol_net.cuda()
    pol_optim = torch.optim.Adam(pol_net.parameters())
    pol_loss_fn = torch.nn.MSELoss()

    im_net = ImPredictor(gp)
    im_net = im_net.cuda()
    im_optim = torch.optim.Adam(im_net.parameters())
    im_loss_fn = torch.nn.MSELoss()
    
    for tx in range(epochs):
        pol_losses = []
        im_losses = []
        for ix in range(train_x.shape[0]):
            pol_optim.zero_grad()
            pol_pred = pol_net(train_x[ix:ix+1, -3:])
            pol_loss = pol_loss_fn(pol_pred, train_y[ix:ix+1])
            pol_loss.backward()
            pol_optim.step()
            pol_losses.append(pol_loss.item())

            im_optim.zero_grad()
            im_pred = im_net(train_x[ix:ix+1, :-3])
            im_loss = im_loss_fn(im_pred, train_y[ix:ix+1])
            im_loss.backward()
            im_optim.step()
            im_losses.append(im_loss.item())
        print('Loss (%d):' % tx, np.mean(pol_losses), np.mean(im_losses))
    
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
                if np.linalg.norm(pol-policy_params) < 0.01:
                    duplicate = True
            if not duplicate:
                added.append(policy_params)
                valid[-1].append(entry)
            else:
                found += 1
    print('Duplicates Found:', found)
    return valid


CUDA = True
MECH_TYPE = 'img'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='')
    parser.add_argument('--pretrained-nn', type=str, required=True)
    parser.add_argument('--L', required=True, type=int)
    args = parser.parse_args()
    print(args)

    #  Load dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    results = [bb[::] for bb in raw_results[:args.L]]  
    results = remove_duplicates(results)
    results = [item for sublist in results for item in sublist][::2]
    data = parse_pickle_file(results)
    
    val_results = [bb[::] for bb in raw_results[81:]]
    val_results = [item for sublist in val_results for item in sublist]
    # print(val_results[0].mechanism_params)
    val_data = parse_pickle_file(val_results)
    
    train_set = setup_data_loaders(data=data, batch_size=16, single_set=True)
    val_set = setup_data_loaders(data=val_data, batch_size=16, single_set=True)
    
    # Extract features for this dataset.
    print('Extracting Features')
    extractor = FeatureExtractor(pretrained_nn_path=args.pretrained_nn)
    if CUDA:
        extractor.cuda()
    train_x, train_y = extract_feature_dataset(train_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)

    mu = torch.mean(train_x, dim=0, keepdims=True)
    std = torch.std(train_x, dim=0, keepdims=True)+1e-3
    print(std)
    train_xs = ((train_x-mu)/std).detach().clone()
    print('Features Extracted')
    print(train_x.shape, train_y.shape)
    val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)
    val_x = (val_x - mu)/std
    print('Data Size:', train_x.shape, train_y.shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-8, 6.25e-6))  # 6.25e-6
    gp = ProductDistanceGP(train_x=train_xs,
                           train_y=train_y,
                           likelihood=likelihood,
                           mech_encoding=MECH_TYPE)
    if CUDA:
        likelihood = likelihood.cuda()
        gp = gp.cuda()
    gp.train()
    likelihood.train()
    # optimizer = torch.optim.Adam([{'params': gp.im_net.parameters(), 'lr': 1e-3},
    #                               {'params': gp.pol_net.parameters(), 'lr': 1e-3},
    #                               {'params': gp.covar_module.parameters(), 'lr': 1e-3},
    #                               {'params': gp.likelihood.parameters(), 'lr': 1e-3}])
    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3}])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    val_loss_fn = torch.nn.MSELoss()
    best = 1000

    # pretrain_modules(gp, train_x, train_y)

    with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
        for tx in range(50000):
            optimizer.zero_grad()
            output = gp(train_xs)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            if tx % 100 == 0:
                print(tx, loss.item(), gp.likelihood.noise.item(), gp.covar_module.base_kernel.lengthscale, gp.covar_module.outputscale.item())
                #gp.eval()
                #likelihood.eval()
                #output = gp(val_x)
                #val_loss = val_loss_fn(output.mean, val_y)
                #output = gp(train_xs)
                #train_loss = val_loss_fn(output.mean, train_y)
                #print('Train Loss:', train_loss.item())
                #print('Val Loss:', val_loss.item())
                #gp.train()
                #likelihood.train()
                
                #if loss.item() < best:
            
            if tx % 50 == 0 and loss.item() < best:
                print('Saving to', args.save_path)
                torch.save((gp.state_dict(), train_xs, train_y, mu, std), args.save_path)
                best = loss.item()
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

