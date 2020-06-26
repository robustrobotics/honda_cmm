"""
This file is meant to check the calibration of a predictor. 
Are the uncertainties representative of a test set.
https://arxiv.org/pdf/1807.00263.pdf
"""
from collections import defaultdict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from learning.modules.policy_encoder import PolicyEncoder
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as SpatialEncoder
from learning.modules.image_encoder import ImageEncoder as CNNEncoder
from learning.models.nn_disp_pol_vis import DistanceRegressor
from learning.dataloaders import setup_data_loaders, parse_pickle_file, PolicyDataset
from learning.models.nn_with_kernel import FeatureExtractor, DistanceGP, ProductDistanceGP
import gpytorch
from gpytorch.kernels import GridInterpolationKernel, ScaleKernel, RBFKernel
from utils.util import load_model, read_from_file
from learning.gp.viz_doors import viz_3d_plots, _true_callback
from utils import util
import numpy as np
import matplotlib.pyplot as plt
from learning.models.nn_with_kernel import extract_feature_dataset


def plot_calibration(likelihood, gp, val_x, val_y):
    intervals = defaultdict(int)
    ps = [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    total = 0.
    with gpytorch.settings.max_cg_iterations(10000), gpytorch.settings.max_preconditioner_size(200), gpytorch.settings.fast_computations(solves=False, log_prob=False, covar_root_decomposition=False):
        for ix in range(0, val_x.shape[0]):
            pred = likelihood(gp(val_x[ix:ix+1, :]))
            lower, upper = pred.confidence_region()
            lower = lower.cpu().detach().numpy()
            upper = upper.cpu().detach().numpy()
            true = val_y[ix].item()
            norm = torch.distributions.normal.Normal(
                        pred.loc, 
                        torch.sqrt(pred.variance))
            print(true, pred.loc)
            if torch.isnan(norm.cdf(true)):
                continue
            total += 1
            for p in ps:
                if norm.cdf(true) <= p:
                    intervals[p] += 1.
                

    for k, v in intervals.items():
        print(k, v, total)
        plt.scatter(k, v/total)

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k-')
    plt.savefig('cal.png')


def plot_door_sim(likelihood, gp, raw_data, train_xs, mu, std):
    # Grab one entry for each mechanism.
    doors_datasets = []
    val_results = [res[0] for res in raw_data]
    val_data = parse_pickle_file(val_results)
    val_set  = setup_data_loaders(data=val_data, batch_size=16, single_set=True)
    val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)
    val_x = (val_x - mu)/std
    doors_datasets.append((val_x, val_y))
    
    train_embed = gp.embed_input(train_xs)
    val_embed = gp.embed_input(val_x)

    # all_data = torch.cat([train_embed, val_embed], 0)
    # all_data = all_data - all_data[:train_embed.shape[0], :].min(0)[0]
    # all_data = 2*(all_data/all_data[:train_embed.shape[0], :].max(0)[0]) - 1

    # train_embed = all_data[:train_embed.shape[0], :]
    # val_embed = all_data[train_embed.shape[0]:, :]



    # val_embed = val_embed - all_data.min(0)[0]
    # train_embed = train_embed - train_embed.min(0)[0]
    # val_embed = 2*(val_embed/train_embed.max(0)[0])-1
    # train_embed = 2*(train_embed/train_embed.max(0)[0])-1
    # val_embed = train_embed - train_embed.min(0)[0]
    # val_embed = 2*(train_embed/train_embed.max(0)[0])-1


    M = val_embed.shape[0]
    sim_mat = np.zeros((M, M))

    lengthscales = gp.covar_module.base_kernel.lengthscale[0, 3:]

    for ix in range(M):
        for jx in range(M):
            w1 = val_embed[ix, 3:]
            w2 = val_embed[jx, 3:]
            print('Train:', train_embed[ix,2:].cpu().detach().numpy())
            print(w1.detach().cpu().numpy(), w2.detach().cpu().numpy())
            print(w1.shape, w2.shape, lengthscales.shape)
            diff = (w1-w2)/lengthscales
            diff = diff.detach().cpu().numpy()
            sim_mat[ix, jx] = np.exp(-0.5*np.linalg.norm(diff))
    
    plt.matshow(sim_mat)
    plt.savefig('scratch/mat_doors.png')

def plot_action_sim(likelihood, gp, raw_data, train_xs, mu, std, n_doors=10):
    # Grab one entry for each mechanism.
    doors_datasets = []
    for ix in range(len(raw_data)):
        val_results = raw_data[ix][::4]
        val_data = parse_pickle_file(val_results)
        val_set  = setup_data_loaders(data=val_data, batch_size=16, single_set=True)
        val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)
        val_x = (val_x - mu)/std
        doors_datasets.append((val_x, val_y))

    # Save images
    for ix in range(0, n_doors):
        util.imsave(raw_data[ix][0].image_data, 'scratch/door_%d.png' % ix)
    
    doors = [doors_datasets[ix][0] for ix in range(n_doors)]
    val_x = torch.cat(doors, axis=0)
    
    train_embed = gp.embed_input(train_xs)
    val_embed = gp.embed_input(val_x)
    # val_embed = val_embed - train_embed.min(0)[0]
    # val_embed = 2*(val_embed/train_embed.max(0)[0])-1

    # all_data = torch.cat([train_embed, val_embed], 0)
    # all_data = all_data - all_data[:train_embed.shape[0], :].min(0)[0]
    # all_data = 2*(all_data/all_data[:train_embed.shape[0], :].max(0)[0]) - 1

    # train_embed = all_data[:train_embed.shape[0], :]
    # val_embed = all_data[train_embed.shape[0]:, :]

    M = val_embed.shape[0]

    sim_mat = np.zeros((M, M))

    lengthscales = gp.covar_module.base_kernel.lengthscale[0, :]

    for ix in range(M):
        for jx in range(M):
            w1 = val_embed[ix, :]
            w2 = val_embed[jx, :]
            diff = (w1-w2)/lengthscales
            diff = diff.detach().cpu().numpy()
            sim_mat[ix, jx] = np.exp(-0.5*np.linalg.norm(diff))
    
    plt.matshow(sim_mat)
    plt.savefig('scratch/mat.png')

def plot_image_features(extractor, raw_data):
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    val_results = [bb[0] for bb in raw_results[80:]]
    val_data = parse_pickle_file(val_results)
    val_set  = setup_data_loaders(data=val_data, batch_size=1, single_set=True)
    
    for nx, (_, _, img, _, _) in enumerate(val_set):
        if torch.cuda.is_available():
            img = img.cuda()
        _, points = extractor.pretrained_model.image_module(img)
        points = points.detach().cpu().numpy()[0,:,:]

        img = img[0,:,:,:]
        c, h, w = img.shape
        img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().numpy()

        fig, axes = plt.subplots(1, 1)
        fig.subplots_adjust(hspace=0, wspace=0.1, top=0.9, bottom=0.1)

        axes.imshow(np.transpose(npimg, (1, 2, 0)))

        cmap = plt.get_cmap('viridis')

        for ix in range(points.shape[0]):
            axes.scatter((points[ix, 0]+1)/2.0*w, (points[ix, 1]+1)/2.0*h, s=5, c=cmap(ix/points.shape[0]))


        plt.savefig('scratch/door_feats_%d.png' % nx)
        print('Saved', nx)





CUDA = True
MECH_TYPE = 'img'
if __name__ == '__main__':
    gp_model = '/home/mnosew/workspace/honda_cmm/gp_models/gp_50L_100M_factored_nodropout.pt'
    pretrained_model = '/home/mnosew/workspace/honda_cmm/pretrained_models/doors/model_20L_100M.pt'
    print(gp_model)
    # Load the model.
    gp_state, train_xs, train_ys, mu, std = torch.load(gp_model)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 6.25e-6))
    print(train_xs.shape)
    gp = ProductDistanceGP(train_x=train_xs,
                           train_y=train_ys,
                           likelihood=likelihood,
                           mech_encoding=MECH_TYPE)
    extractor = FeatureExtractor(pretrained_nn_path=pretrained_model)
    if CUDA:
        extractor.cuda()
        likelihood.cuda()
        gp.cuda()
    gp.load_state_dict(gp_state)

    # Load validation dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    # val_results = [bb[::] for bb in raw_results[80:]]
    # val_results = [item for sublist in val_results for item in sublist]
    # val_data = parse_pickle_file(val_results)
    # val_set  = setup_data_loaders(data=val_data, batch_size=16, single_set=True)
    # val_x, val_y = extract_feature_dataset(val_set, extractor, use_cuda=CUDA, mech_encoding=MECH_TYPE)
    # val_x = (val_x - mu)/std
    
    gp.eval()
    likelihood.eval()
    
    # plot_image_features(extractor, raw_results)
    plot_door_sim(likelihood, gp, raw_results[80:], train_xs, mu, std)
    plot_action_sim(likelihood, gp, raw_results[80:], train_xs, mu, std)

    # TODO: Plot calibration.
    # plot_calibration(likelihood, gp, val_x, val_y)