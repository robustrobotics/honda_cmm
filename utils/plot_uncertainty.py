import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from learning.modules.policy_encoder import PolicyEncoder
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as SpatialEncoder
from learning.modules.image_encoder import ImageEncoder as CNNEncoder
from learning.models.nn_disp_pol_vis import DistanceRegressor
from learning.dataloaders import setup_data_loaders, parse_pickle_file, PolicyDataset
from learning.models.nn_with_kernel import FeatureExtractor, ProductDistanceGP
import gpytorch
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel, RBFKernel
from utils.util import load_model, read_from_file
from learning.gp.viz_doors import viz_3d_plots, _true_callback
from utils import util
import numpy as np


def get_callback(gp_model, likelihood, feature_extractor, p_mu, p_std):
    def callback(xs, bb_result):
        """ 
        :param xs: A list of (ro, pitch, ra, q) policy tuples.
        :param bb_result: A result object containing the current BB.
        :return: (ys, std, None)
        """
        data = []
        for ro, pitch, radius, q in xs:
            data.append({'type': 'Revolute',
                         'params': [pitch, radius, q],
                         'goal_config': q,
                         'image': bb_result.image_data,
                         'y': 0.,
                         })
        dataset = PolicyDataset(data)
        nn_ys, nn_std = [], []
        for i in range(len(dataset.items)):
            policy_type = dataset.items[i]['type']
            policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
            policy_tensor = dataset.tensors[i].unsqueeze(0)
            image_tensor = dataset.images[i].unsqueeze(0)
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            image_tensor = image_tensor.cuda()
            features = feature_extractor.forward(policy_type_tensor,
                                                 image_tensor,                 
                                                 policy_tensor)
            features = (features - p_mu)/p_std

            #print(gp_model.embed_input(features))
            with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000),  gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
            #with gpytorch.settings.fast_pred_var():
                pred = gp_model(features)
            pred_motion_float = pred.mean.cpu().detach().numpy()[0]
            pred_std = pred.stddev.cpu().detach().numpy()[0]
            nn_ys += [pred_motion_float]
            nn_std += [pred_std]
            # print(pred_std)
        ys = np.array(nn_ys)
        std = np.array(nn_std)
        return ys, std, None

    return callback

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
    parser.add_argument('--gp-path', type=str, required=True)
    parser.add_argument('--pretrained-nn', type=str, required=True)
    args = parser.parse_args()
    print(args)
    CUDA = True

    # Load the GP and Feature Extractor.
    gp_state, train_xs, train_ys, mu, std = torch.load(args.gp_path)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1.e-6))
    gp = ProductDistanceGP(train_x=train_xs,
                    train_y=train_ys,
                    likelihood=likelihood)
    extractor = FeatureExtractor(pretrained_nn_path=args.pretrained_nn)
    if CUDA:
        extractor.cuda()
        likelihood.cuda()
        gp.cuda()
    gp.load_state_dict(gp_state)
    print(gp.covar_module.base_kernel.lengthscale)
    print(gp.likelihood.noise)
    # Load validation dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    #raw_results = read_from_file('100_eval_doors.pickle')
    #val_results = [bb[::] for bb in raw_results]
    #val_results = [item for sublist in val_results for item in sublist]
    val_results = [bb[::] for bb in raw_results[80:]]
    val_results = [item for sublist in val_results for item in sublist]
    val_data = parse_pickle_file(val_results)
    val_set  = setup_data_loaders(data=val_data, batch_size=16, single_set=True)
    val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA)
    val_x = (val_x - mu)/std
    
    gp.eval()
    #gp.set_train_data(train_xs[::10,:], train_ys[::10], strict=False)
    likelihood.eval()
    print('Val Predictions')
    with gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.fast_computations(log_prob=False, solves=False, covar_root_decomposition=False):
        for ix in range(0, 20):#val_x.shape[0]):
            #print(gp.likelihood.noise)
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
    
        #print('COMP:',(upper-lower)/4., pred.stddev.cpu().detach().numpy())
    #viz_interaction()
    viz_3d_plots(xs=[],
                 callback=get_callback(gp, likelihood, extractor, mu, std),
                 bb_result=val_results[0],
                 n_rows=1)
