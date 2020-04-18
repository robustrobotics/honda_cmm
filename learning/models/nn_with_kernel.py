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


class DistanceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, pretrained_nn_path=''):
        super(DistanceGP, self).__init__(train_x, train_y, likelihood)
        
        # Load the pretrained NN from the given file.

        # Create a mean function. 
        # Note: Should this be the pretrained NN prediction?
        self.mean_module = gpytorch.means.ConstantMean()
        # Note: Should probably make kernel dims smaller by adding a learned Linear layer on top?
        self.covar_module = GridInterpolationKernel(
                                ScaleKernel(RBFKernel(ard_num_dims=2)),
                                num_dims=2, grid_size=100)
        self.pretrained_model = load_model(model_fname=pretrained_nn_path,
                                           hdim=16)
        self.lin = nn.Linear(32, 2)
        print(self.covar_module.grid[0].dtype) 
    def feature_extractor(self, x):
        policy_type = 1
        theta = x[:, :3]
        im = x[:, 3:].reshape(x.shape[0], 3, 118, 116)
        if policy_type == 0:
            policy_type = 'Prismatic'
        else:
            policy_type = 'Revolute'
        pol = self.pretrained_model.policy_modules[policy_type].forward(theta)
        im, points = self.pretrained_model.image_module(im)

        x = torch.cat([pol, im], dim=1)
        x = F.relu(self.pretrained_model.fc1(x))
        x = F.relu(self.pretrained_model.fc2(x))
        x = self.lin(x)
        print('Computed features')
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x - x.min(0)[0]
        x = 2*(x/x.max(0)[0])-1
        print('X:', x.shape)
        mean_x = self.mean_module(x)
        print('Got Mean')
        covar_x = self.covar_module(x)
        print('Got Covar')
        print(mean_x.shape, covar_x)
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

if __name__ == '__main__':
    #  Load dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    results = [bb[::50] for bb in raw_results]  # For now just grab every 10th interaction with each bbb.
    results = [item for sublist in results for item in sublist]

    data = parse_pickle_file(results)
    train_set, val_set, _ = setup_data_loaders(data=data, batch_size=16)
    train_x, train_y = convert_training_data(train_set)
    val_x, val_y = convert_training_data(val_set)
    print('Data Size:', train_x.shape, train_y.shape)
    print(train_x.size(0), train_y.size(0))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = DistanceGP(train_x=train_x,
                    train_y=train_y,
                    likelihood=likelihood,
                    pretrained_nn_path='/home/mnosew/workspace/honda_cmm/pretrained_models/doors/model_100L_100M.pt')
    gp.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': gp.covar_module.parameters()},
                                  {'params': gp.lin.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    
    for _ in range(10):
        optimizer.zero_grad()
        output = gp(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        print(loss.item())
    gp.eval()
    print(val_x[0:1, :].shape)
    pred = gp(val_x[0:1, :])
    print(pred, pred.confidence_region(), val_y[0])
