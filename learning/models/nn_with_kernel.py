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
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel, RBFKernel
from utils.util import load_model, read_from_file


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_nn_path):
        super(FeatureExtractor, self).__init__()
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

        x = torch.cat([pol, im], dim=1)
        x = F.relu(self.pretrained_model.fc1(x))
        x = F.relu(self.pretrained_model.fc2(x))
        return x

    def forward_cached(self, policy_type, im, theta):
        policy_type = 'Revolute'
        pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        x = torch.cat([pol, im], dim=1)
        x = F.relu(self.pretrained_model.fc1(x))
        x = F.relu(self.pretrained_model.fc2(x))
        return x


class DistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(DistanceGP, self).__init__(train_x, train_y, likelihood)
        num_gp_dims = 2 
        # Create a mean function. 
        # Note: Should this be the pretrained NN prediction?
        self.mean_module = gpytorch.means.ConstantMean()
        # Note: Should probably make kernel dims smaller by adding a learned Linear layer on top?
        #self.covar_module = GridInterpolationKernel(
        #                        ScaleKernel(RBFKernel(ard_num_dims=num_gp_dims)),
        #                        num_dims=num_gp_dims, grid_size=200)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=num_gp_dims, lengthscale_constraint=gpytorch.constraints.Interval(0.0001, 1.)))
        self.lin = nn.Linear(32, num_gp_dims)

    def forward(self, x, disp=False):
        x = self.lin(x)
        #print(x.min(0)[0], x.max(0)[0])
        x = x - x.min(0)[0]
        x = 2*(x/x.max(0)[0])-1
        if disp:
            print(x.shape)
            print(x)
        #x += torch.randn(x.shape).cuda()*1e-2
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

def extract_feature_dataset(dataset, extractor, use_cuda=False):
    """ First extract the features from the dataset."""
    features, ys = [], []
    for policy_type, theta, im, y, _ in dataset:
        if use_cuda:
            theta = theta.cuda()
            im = im.cuda()
            y = y.cuda()
        feats = extractor(policy_type, im, theta) 
        #feats += torch.randn(feats.shape).cuda()*1e-3
        features.append(feats.detach())
        ys.append(y)
    return torch.cat(features, dim=0), torch.cat(ys, dim=0).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='')
    parser.add_argument('--pretrained-nn', type=str, required=True)
    args = parser.parse_args()
    print(args)
    # '/home/mnosew/workspace/honda_cmm/pretrained_models/doors/model_80L_100M.pt'
    CUDA = True
    #  Load dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    results = [bb[::] for bb in raw_results[:80]]  # For now just grab every 10th interaction with each bbb.
    val_results = [bb[::] for bb in raw_results[80:]]
    results = [item for sublist in results for item in sublist]
    val_results = [item for sublist in val_results for item in sublist]
    data = parse_pickle_file(results)
    val_data = parse_pickle_file(val_results)
    train_set, _, _ = setup_data_loaders(data=data, batch_size=16)
    val_set, _, _ = setup_data_loaders(data=val_data, batch_size=16)
    extractor = FeatureExtractor(pretrained_nn_path=args.pretrained_nn)
    if CUDA:
        extractor.cuda()
    print('Extracting Features')
    train_x, train_y = extract_feature_dataset(train_set, extractor, use_cuda=CUDA)
    mu = torch.mean(train_x, dim=0, keepdims=True)
    std = torch.std(train_x, dim=0, keepdims=True)+1e-3
    train_xs = ((train_x-mu)/std).detach().clone()
    print('Features Extracted')
    print(train_x.shape, train_y.shape)
    val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA)
    val_x = (val_x - mu)/std
    del extractor, train_set, val_set
    print('Data Size:', train_x.shape, train_y.shape)
    print(train_x.size(0), train_y.size(0))
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.0001))  # 0.000025
    gp = DistanceGP(train_x=train_xs,
                    train_y=train_y,
                    likelihood=likelihood)
    if CUDA:
        likelihood = likelihood.cuda()
        gp = gp.cuda()
    gp.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': gp.covar_module.parameters()},
                                  {'params': gp.lin.parameters()},
                                  {'params': gp.likelihood.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    #with gpytorch.settings.max_cg_iterations(2000):
    for _ in range(500):
        optimizer.zero_grad()
        output = gp(train_xs)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        print(loss.item(), gp.likelihood.noise.item(), gp.covar_module.base_kernel.lengthscale, gp.covar_module.outputscale)
    if args.save_path != '':
        torch.save((gp.state_dict(), train_xs, train_y, mu, std), args.save_path)
         
    gp.eval()
    print('Val Predictions')
    for ix in range(0, 20):#val_x.shape[0]):
        pred = likelihood(gp(val_x[ix:ix+1, :]))
        lower, upper = pred.confidence_region()
        lower = lower.cpu().detach().numpy()
        upper = upper.cpu().detach().numpy()
        true = val_y[ix].item()
        
        p = pred.mean.cpu().detach().numpy()[0]
        if p < lower or p > upper:
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

