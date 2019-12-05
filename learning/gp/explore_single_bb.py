from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum, ConstantKernel, ExpSineSquared
import numpy as np
from gen.generator_busybox import create_simulated_baxter_slider
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from argparse import Namespace
from scipy.optimize import minimize
from utils import util
from utils.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
from actions import policies
import operator
import pybullet as p
from actions.policies import Prismatic, Revolute
import torch
from learning.dataloaders import PolicyDataset, parse_pickle_file
from gen.generate_policy_data import generate_dataset
from collections import namedtuple
import time
from actions.policies import Policy, generate_policy, Revolute, Prismatic
import itertools
from functools import reduce

BETA = 5 #5.0
# takes in an x and plot variables and returns the values to be plotted
def get_point_from_x(x, linear_param, angular_param, policy_type):
    point = np.zeros(2)
    linear_name = linear_param.param_name
    angular_name = angular_param.param_name
    if policy_type == 'slider':
        if linear_name == 'config':
            point[1] = abs(x[-1])
        if angular_name == 'pitch':
            if x[-1] < 0:
                point[0] = x[0] - np.pi
            else:
                point[0] = x[0]
        if angular_name == 'yaw':
            point[0] = x[1]
    if policy_type == 'door':
        if linear_name == 'config':
            point[1] = x[-1]
        if linear_name == 'radius':
            point[1] = x[2]
        if angular_name == 'pitch':
            point[0] = x[1]
        if angular_name == 'roll':
            point[0] = x[0]
    return point

# takes in an optimization x and returns a policy
def get_policy_from_x(type, x, mech):
    if type == 'Revolute':
        rot_axis_roll = x[0]

        if not mech.flipped:
            rot_axis_pitch = np.pi
        else:
            rot_axis_pitch = 0.0
        rot_axis_world = util.quaternion_from_euler(rot_axis_roll, rot_axis_pitch, 0.0)
        #radius_x = x[1]
        radius_x = mech.get_radius_x()
        radius = [-radius_x, 0.0, 0.0]
        p_handle_base_world = mech.get_pose_handle_base_world().p
        p_rot_center_world = p_handle_base_world + util.transformation(radius, [0., 0., 0.], rot_axis_world)
        rot_orn = [0., 0., 0., 1.]
        return Revolute(p_rot_center_world,
                        rot_axis_roll,
                        rot_axis_pitch,
                        rot_axis_world,
                        radius_x,
                        rot_orn)
                        #delta_roll,
                        #delta_pitch,
                        #delta_radius_x)
    if type == 'Prismatic':
        pitch = x[0]
        yaw = 0.0
        pos = mech.get_pose_handle_base_world().p
        orn = [0., 0., 0., 1.]
        return Prismatic(pos, orn, pitch, yaw)

# takes in a result and returns an x
def get_x_from_result(result):
    if result.policy_params.type == 'Prismatic':
        pitch = result.policy_params.params.pitch
        yaw = result.policy_params.params.yaw
        q = result.config_goal
        return [pitch, yaw, q]
    elif result.policy_params.type == 'Revolute':
        axis_roll = result.policy_params.params.rot_axis_roll
        axis_pitch = result.policy_params.params.rot_axis_pitch
        radius_x = result.policy_params.params.rot_radius_x
        q = result.config_goal
        return [axis_roll, axis_pitch, radius_x, q]

# takes in a policy and returns and optimization x and the variable bounds
def get_reduced_x_and_bounds(policy_type, policy_params, q, policy_data):
    for policy_param in policy_data:
        if policy_param.param_name == 'pitch':
            pitch_bounds = policy_param.range
        #if policy_param.param_name == 'yaw':
        #    yaw_bounds = policy_param.range
        if policy_param.param_name == 'config':
            config_bounds = policy_param.range
        if policy_param.param_name == 'roll':
            roll_bounds = policy_param.range
        #if policy_param.param_name == 'radius':
        #    radius_bounds = policy_param.range
    if policy_type == 'Prismatic':
        return np.concatenate([[policy_params.params.pitch], [q]]), \
                [pitch_bounds, config_bounds]
    elif policy_type == 'Revolute':
        return np.concatenate([[policy_params.params.rot_axis_roll], [q]]), \
                [roll_bounds, config_bounds]
        #return np.concatenate([[policy_params.params.rot_axis_roll, \
        #                            policy_params.params.rot_radius_x], [q]]), \
        #        [roll_bounds, radius_bounds, config_bounds]

# this takes in an optimization x and returns the tensor of just the policy
# params
def get_policy_tensor(policy_type, x, mech):
    policy_list = [get_policy_list(policy_type, x, mech)[0][:-1]]
    if policy_type == 'Prismatic':
        return torch.tensor(policy_list).float()  # hard code yaw to be 0
    elif policy_type == 'Revolute':
        return torch.tensor(policy_list).float()

# this takes in an optimization x and returns an x with the policy params
def get_policy_list(policy_type, x, mech):
    if policy_type == 'Prismatic':
        return [[x[0], 0.0, x[-1]]]
    elif policy_type == 'Revolute':
        if not mech.flipped:
            pitch = np.pi
        else:
            pitch = 0.0
        radius = mech.get_radius_x()
        return [[x[0], pitch, radius, x[-1]]]

def process_data(data, n_train):
    """
    Takes in a dataset in our typical format and outputs the dataset to fit the GP.
    :param data:
    :return:
    """
    xs, ys = [], []
    for entry in data[0:n_train]:
        x = get_x_from_result(entry)
        xs.append(x)
        ys.append(entry.net_motion)

    X = np.array(xs)
    Y = np.array(ys).reshape(-1, 1)
    '''
    if max_dist:
        x_preds = []
        for theta in np.linspace(-np.pi, 0, num=100):
            x_preds.append([theta, 0, max_dist])
        X_pred = np.array(x_preds)
        return X, Y, X_pred
    else:
    '''
    return X, Y

def get_nn_preds(results, model, ret_dataset=False, use_cuda=False):
    data = parse_pickle_file(results)
    dataset = PolicyDataset(data)
    pred_motions = []
    for i in range(len(dataset.items)):
        policy_type = dataset.items[i]['type']
        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
        policy_tensor = dataset.tensors[i].unsqueeze(0)
        config_tensor = dataset.configs[i].unsqueeze(0)
        image_tensor = dataset.images[i].unsqueeze(0)
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            config_tensor = config_tensor.cuda()
            image_tensor = image_tensor.cuda()
        pred_motion, _ = model.forward(policy_type_tensor,
                                       policy_tensor,
                                       config_tensor,
                                       image_tensor)
        if use_cuda:
            pred_motion_float = pred_motion.cpu().detach().numpy()[0][0]
        else:
            pred_motion_float = pred_motion.detach().numpy()[0][0]
        pred_motions += [pred_motion_float]
    if ret_dataset:
        return pred_motions, dataset
    else:
        return pred_motions

def objective_func(x, policy_type, image_tensor, model, use_cuda):
    policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
    x = x.squeeze()
    policy_tensor = torch.tensor([[x[0], 0.0]]).float() # hard code yaw to be 0
    config_tensor = torch.tensor([[x[-1]]]).float()
    if use_cuda:
        policy_type_tensor = policy_type_tensor.cuda()
        policy_tensor = policy_tensor.cuda()
        config_tensor = config_tensor.cuda()
        image_tensor = image_tensor.cuda()
    val = -model.forward(policy_type_tensor, policy_tensor, config_tensor, image_tensor)
    if use_cuda:
        val = val.cpu()
    val = val.detach().numpy()
    val = val.squeeze()
    return val

def test_model(sampler, args):
    """
    Maximize the GP mean function to get the best policy.
    :param sampler: A GP fit to the current BusyBox.
    :return: Regret.
    """
    # Optimize the GP to get the best policy.
    ucb = False
    policy, q, start_policy, start_q = sampler.optim.optimize_gp(ucb)

    # Execute the policy and observe the true motion.
    debug = False
    viz = False
    no_gripper = True
    _, gripper = setup_env(sampler.bb, viz, debug, no_gripper)
    pose_handle_base_world = sampler.mech.get_pose_handle_base_world()
    traj = policy.generate_trajectory(pose_handle_base_world, q, debug=debug)
    _, motion, _ = gripper.execute_trajectory(traj, sampler.mech, policy.type, debug=debug)

    # Calculate the regret.
    max_d = sampler.mech.get_max_dist()
    regret = (max_d - motion)/max_d

    # Get the initial and final x from the optimization.
    start_result = util.Result(start_policy.get_policy_tuple(), None, 0.0, 0.0,
                           None, None, start_q, None, None, 1.0, None)
    start_x = get_x_from_result(start_result)
    stop_result = util.Result(policy.get_policy_tuple(), None, 0.0, 0.0,
                           None, None, q, None, None, 1.0, None)
    stop_x = get_x_from_result(stop_result)
    return regret, start_x, stop_x

class GPOptimizer(object):

    def __init__(self, urdf_num, bb, image_data, n_samples, beta, gp, nn=None):
        """
        Initialize one of these for each BusyBox.
        """
        self.sample_policies = []
        self.nn_samples = []
        self.nn = nn
        self.mech = bb._mechanisms[0]
        self.beta = beta
        self.gp = gp

        # Generate random policies.
        for _ in range(n_samples):
            # Get the mechanism from the dataset.
            # TODO: change random_policies to True when ready
            random_policy = generate_policy(bb, self.mech, False, 1.0)
            q = random_policy.generate_config(self.mech, None)
            policy_type = random_policy.type
            policy_tuple = random_policy.get_policy_tuple()

            results = [util.Result(policy_tuple, None, 0.0, 0.0,
                                   None, None, q, image_data, None, 1.0, True)]
            self.sample_policies.append(results)

            if not self.nn is None:
                nn_preds, self.dataset = get_nn_preds(results, nn, ret_dataset=True, use_cuda=False)
                self.nn_samples.append(nn_preds)
            else:
                self.dataset = None
                self.nn_samples.append(None)

    def _optim_result_to_torch(self, policy_type, x, image_tensor, use_cuda=False):
        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
        policy_tensor = get_policy_tensor(policy_type, x, self.mech)
        config_tensor = torch.tensor([[x[-1]]]).float()
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            config_tensor = config_tensor.cuda()
            image_tensor = image_tensor.cuda()

        return [policy_type_tensor, policy_tensor, config_tensor, image_tensor]

    def _objective_func(self, x, policy_type, ucb, image_tensor=None):
        x = x.squeeze()

        X = np.array(get_policy_list(policy_type, x, self.mech))
        Y_pred, Y_std = self.gp.predict(X, return_std=True)

        if not self.nn is None:
            inputs = self._optim_result_to_torch(policy_type, x, image_tensor, False)
            val, _ = self.nn.forward(*inputs)
            val = val.detach().numpy()
            Y_pred += val.squeeze()

        if ucb:
            obj = -Y_pred[0] - np.sqrt(self.beta) * Y_std[0]
        else:
            obj = -Y_pred[0]

        return obj

    def _get_pred_motions(self, data, ucb, nn_preds=None):
        X, Y = process_data(data, len(data))
        y_pred, y_std = self.gp.predict(X, return_std=True)

        if not nn_preds is None:
            y_pred += np.array(nn_preds)

        if ucb:
            return y_pred + np.sqrt(self.beta) * y_std, self.dataset
        else:
            return y_pred, self.dataset

    def optimize_gp(self, ucb):
        """
        Find the input (policy) that maximizes the GP (+ NN) output.
        :param ucb: If True use the GP-UCB criterion
        :return: x_final, the optimal policy according to the current model.
        """
        samples = []

        # Generate random policies.
        for res, nn_preds in zip(self.sample_policies, self.nn_samples):
            # Get predictions from the GP.
            sample_disps, dataset = self._get_pred_motions(res, ucb, nn_preds=nn_preds)

            samples.append(((res[0].policy_params.type,
                             res[0].policy_params,
                             res[0].config_goal),
                             sample_disps[0]))

        # Find the sample that maximizes the distance.
        (policy_type_max, params_max, q_max), max_disp = max(samples, key=operator.itemgetter(1))

        # Start optimization from here.
        if self.nn is None:
            images = None
        else:
            images = dataset.images[0].unsqueeze(0)
        policy_data = Policy.get_plot_data(self.mech)
        #print('MAX', params_max)
        x0, bounds = get_reduced_x_and_bounds(policy_type_max, params_max, q_max, policy_data)
        start_policy = get_policy_from_x(policy_type_max, x0, self.mech)
        start_q = q_max
        opt_res = minimize(fun=self._objective_func, x0=x0, args=(policy_type_max, ucb, images),
                       method='L-BFGS-B', options={'eps': 10**-3}, bounds=bounds)
        x_final = opt_res['x']
        stop_policy = get_policy_from_x(policy_type_max, x_final, self.mech)
        stop_q = x_final[-1]
        return stop_policy, stop_q, start_policy, start_q


class UCB_Interaction(object):

    def __init__(self, bb, image_data, plot, args, nn_fname=''):
        # Pretrained Kernel (for Sliders)
        # kernel = ConstantKernel(0.005, constant_value_bounds=(0.005, 0.005)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e2))
        # Pretrained Kernel (for Doors)
        # 0.0202**2 * RBF(length_scale=[0.0533, 0.000248, 0.0327, 0.0278]) + WhiteKernel(noise_level=1e-05)
        self.interaction_data = []
        self.xs, self.ys, self.moves = [], [], []

        self.ix = 0
        self.plot = plot
        self.nn = None
        if nn_fname != '':
            self.nn = util.load_model(nn_fname, args.hdim, use_cuda=False)
        self.bb = bb
        self.image_data = image_data
        self.mech = self.bb._mechanisms[0]
        self.kernel = self.get_kernel()
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                              n_restarts_optimizer=10)
        self.optim = GPOptimizer(args.urdf_num, self.bb, self.image_data, args.n_gp_samples, BETA, self.gp, nn=self.nn)

    def get_kernel(self):
        # TODO: in the future will want the GP to take in all types of policy
        # params, not just the correct type
        noise = 1e-5
        if self.mech.mechanism_type == 'Slider':
            variance = 0.005
            l_pitch = 0.10
            l_yaw = 100
            l_q = 0.1
            return ConstantKernel(variance,
                                    constant_value_bounds=(variance,
                                                            variance)) \
                    * RBF(length_scale=(l_pitch,
                                        l_yaw,
                                        l_q),
                        length_scale_bounds=((l_pitch, l_pitch),
                                            (l_yaw, l_yaw),
                                            (l_q, l_q))) \
                    + WhiteKernel(noise_level=noise,
                                    noise_level_bounds=(1e-5, 1e2))
        elif self.mech.mechanism_type == 'Door':
            variance = 0.005
            l_roll = 0.25
            l_pitch = 100
            l_radius = 100
            l_q = 1.0 # Keep greater than 0.5.
            return ConstantKernel(variance,
                                    constant_value_bounds=(variance,
                                                            variance)) \
                    * RBF(length_scale=(l_roll,
                                        l_pitch,
                                        l_radius,
                                        l_q),
                        length_scale_bounds=((l_roll, l_roll),
                                            (l_pitch, l_pitch),
                                            (l_radius, l_radius),
                                            (l_q, l_q))) \
                    + WhiteKernel(noise_level=noise,
                                    noise_level_bounds=(1e-5, 1e2))
    def sample(self):
        # (1) Choose a point to interact with.
        if len(self.xs) < 1 and self.nn is None:
            # (a) Choose policy randomly.
            policy = generate_policy(self.bb, self.mech, False, 1.0)
            q = policy.generate_config(self.mech, None)
        else:
            # (b) Choose policy using UCB bound.
            ucb = True
            policy, q, _, _ = self.optim.optimize_gp(ucb)
            #print(policy.get_policy_tuple())

        self.ix += 1
        return policy, q

    def update(self, result):
        # TODO: Update without the NN.

        # (3) Update GP.
        x = get_x_from_result(result)
        self.xs.append(x)
        if self.nn is None:
            self.ys.append([result.net_motion])
        else:
            policy_type = result.policy_params.type
            inputs = self.optim._optim_result_to_torch(policy_type,
                                    self.xs[-1],
                                    self.optim.dataset.images[0].unsqueeze(0),
                                    use_cuda=False)
            nn_pred = self.nn.forward(*inputs)[0]
            nn_pred = nn_pred.detach().numpy().squeeze()
            self.ys.append([result.net_motion - nn_pred])

        self.moves.append([result.net_motion])
        self.gp.fit(np.array(self.xs), np.array(self.ys))
        # (4) Visualize GP.
        #if self.ix % 1 == 0 and self.plot:
            #params = mech.get_mechanism_tuple().params
            #print('Range:', params.range/2.0)
            #print('Angle:', np.arctan2(params.axis[1], params.axis[0]))
            #print('GP:', gp.kernel_)

            # #plt.clf()
            # plt.figure(figsize=(15, 15))
            # for x, y in zip(xs, ys):
            #     plt.scatter(x[0], x[2], s=200)
            # plt.title('policy samples')
            # plt.xlabel('pitch')
            # plt.ylabel('q')
            # #plt.savefig('gp_samples_%d.png' % ix)
            # plt.show()
            # viz_gp(gp, result, ix, bb, nn=nn)

    def calc_avg_regret(self):
        regrets = []
        max_dist = self.mech.get_max_dist()
        for y in self.moves:
            regrets.append((max_dist - y[0])/max_dist)
        if len(regrets) > 0:
            return np.mean(regrets)
        else:
            return 'n/a'


def format_batch(type, X_pred, mech, image_data):
    results = []

    for ix in range(X_pred.shape[0]):
        policy = get_policy_from_x(type, X_pred[ix], mech)
        q = X_pred[ix, -1]
        result = util.Result(policy.get_policy_tuple(), None, 0.0, 0.0,
                             None, None, q, image_data, None, 1.0, False)
        results.append(result)

    data = parse_pickle_file(results)
    dataset = PolicyDataset(data)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=len(dataset))
    return train_loader

'''
def viz_gp(gp, result, num, bb, nn=None):
    n_pitch = 10
    # plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(40, 4))

    axes = axes.flatten()
    for ix, pitch in enumerate(np.linspace(-np.pi, 0, n_pitch)):
        x_preds = []
        for q in np.linspace(-0.25, 0.25, num=100):
            x_preds.append([pitch, 0, q])
        X_pred = np.array(x_preds)

        Y_pred, Y_std = gp.predict(X_pred, return_std=True)
        if not nn is None:
            loader = format_batch(X_pred, bb, type)
            k, x, q, im, _, _ = next(iter(loader))
            pol = torch.Tensor([util.name_lookup[k[0]]])
            nn_preds = nn(pol, x, q, im).detach().numpy()
            Y_pred += nn_preds
            axes[ix].plot(X_pred[:, 2], nn_preds[:, 0], c='b', ls='-')

        # print(Y_pred)
        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0], c='r', ls='-')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] + Y_std, c='r', ls='--')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] - Y_std, c='r', ls='--')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] + np.sqrt(40)*Y_std, c='g')
        fs = 8
        axes[ix].set_xlabel('q', fontsize=fs)
        axes[ix].set_ylabel('d', fontsize=fs)
        axes[ix].set_ylim(0, 0.15)
        x0, x1 = axes[ix].get_xlim()
        y0, y1 = axes[ix].get_ylim()
        axes[ix].set_aspect((x1 - x0) / (y1 - y0))
        axes[ix].set(adjustable='box')
        axes[ix].set_title('pitch=%.2f' % pitch, fontsize=fs)
    plt.show()
    # plt.savefig('gp_estimates_tuned_%d.png' % num)
'''

def viz_circles(image_data, mech, sample_points=[], opt_points=[], gp=None, nn=None, bb_i=0, plot_dir_prefix=''):
    policy_plot_data = Policy.get_plot_data(mech)

    n_angular = 40
    n_linear = 20
    N_BINS = 5
    n_params = len(policy_plot_data)

    all_angular_params = list(filter(lambda x: x.type == 'angular', policy_plot_data))
    all_linear_params = list(filter(lambda x: x.type == 'linear', policy_plot_data))

    # make figure of an image of the mechanism
    plt.ion()
    fig, ax = plt.subplots()
    w, h, im = image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax.imshow(np_im)

    # for each pair of (linear, angular) param pairs make a figure
    for angular_param in all_angular_params:
        if angular_param.range[0] == angular_param.range[1]:
            continue
        for linear_param in all_linear_params:
            if linear_param.range[0] == linear_param.range[1]:
                continue
            linear_vals = np.linspace(*linear_param.range, n_linear)
            angular_vals = np.linspace(*angular_param.range, n_angular)
            l, a = np.meshgrid(linear_vals, angular_vals)
            mean_colors = np.zeros(l.shape)

            # bin the other param values
            all_other_params = list(filter(lambda x: (x!=angular_param)
                                                    and (x!=linear_param),
                                            policy_plot_data))
            other_vals_and_inds = []
            for other_params in all_other_params:
                if other_params.range[0] == other_params.range[1]:
                    n_bins = 1
                else:
                    n_bins = N_BINS
                other_vals_and_inds += [list(enumerate(np.linspace(*other_params.range, n_bins)))]

            # TODO: only works for up to 2 other_params (will have to figure out new
            # visualization past that)
            fig = plt.figure()
            plt.suptitle(angular_param.param_name + ' vs ' + linear_param.param_name)
            if len(other_vals_and_inds) == 1:
                n_rows = 1
                n_cols = len(other_vals_and_inds[0])
            elif len(other_vals_and_inds) == 2:
                n_cols = len(other_vals_and_inds[0])
                n_rows = len(other_vals_and_inds[1])

            # for each other value add a dimension of plots to the figure
            for other_val_and_ind in itertools.product(*other_vals_and_inds):
                # make matrix of all values to predict dist for
                X_pred = np.zeros((n_angular*n_linear, n_params))
                xpred_rowi = 0
                for ix in range(0, n_angular):
                    for jx in range(0, n_linear):
                        X_pred[xpred_rowi, angular_param.param_num] = angular_vals[ix]
                        X_pred[xpred_rowi, linear_param.param_num] = linear_vals[jx]
                        for otheri, other_param_val in enumerate(other_val_and_ind):
                            other_param_num = all_other_params[otheri].param_num
                            X_pred[xpred_rowi, other_param_num] = other_param_val[1]
                        xpred_rowi += 1


                Y_pred = np.zeros((X_pred.shape[0]))
                if not gp is None:
                    Y_pred = np.add(Y_pred, gp.predict(X_pred))
                if not nn is None:
                    type = 'Revolute' if mech.mechanism_type == 'Door' else 'Prismatic'
                    loader = format_batch(type, X_pred, mech, image_data)
                    k, x, q, im, _, _ = next(iter(loader))
                    pol = torch.Tensor([util.name_lookup[k[0]]])
                    nn_preds = nn(pol, x, q, im)[0].detach().numpy()
                    Y_pred = np.add(Y_pred, nn_preds.squeeze())
                mean_colors = Y_pred.reshape(n_angular, n_linear)

                row_col = [o[0]+1 for o in other_val_and_ind]
                if len(row_col) == 1:
                    subplot_num = 1*row_col[0]
                else:
                    subplot_num = reduce(lambda x,y: n_cols*(x-1)+y, row_col)
                ax = plt.subplot(n_rows, n_cols, subplot_num, projection='polar')
                ax.set_title('\n'.join([str(all_other_params[other_param_i].param_name) \
                    + ' = ' + str("%.2f" % other_val) for other_param_i, \
                    (plot_i, other_val) in enumerate(other_val_and_ind)]), \
                    fontsize=10)
                max_dist = mech.get_max_dist()
                im = polar_plots(ax, mean_colors, max_dist, angular_param,
                                linear_param, points=sample_points+opt_points,
                                colorbar=False)
            add_colorbar(fig, im)
            plot_dir = 'gp_plots/'
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            if plot_dir_prefix is not '':
                plot_dir += plot_dir_prefix
                if not os.path.isdir(plot_dir):
                    os.mkdir(plot_dir)
            plot_dir += '/bb_%i' % bb_i
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            plot_dir += '/%s' % angular_param.param_name+linear_param.param_name
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)
            plt.savefig(plot_dir+'/%i.png' % (len(sample_points)))

    #plt.show()
    #input('Enter to close plots')
    plt.close('all')

def polar_plots(ax, colors, vmax, angular_param, linear_param, points=None, colorbar=False):
    n_ang, n_lin = colors.shape

    # for slider policies
    if (abs(angular_param.range[0]-angular_param.range[1]) == np.pi) and \
        (linear_param.range[0] == -linear_param.range[1]):
        thp = np.linspace(0, 2*np.pi, n_ang*2)
        rp = np.linspace(0, max(linear_param.range), n_lin//2)
        rp, thp = np.meshgrid(rp, thp)
        cp = np.zeros(rp.shape)

        cp[0:n_ang, :] = colors[:, 0:n_lin//2][:, ::-1]
        cp[n_ang:, :] = np.copy(colors[:, n_lin//2:])
        type = 'slider'
    # for door policies
    else:
        thp = np.linspace(*angular_param.range, n_ang)
        rp = np.linspace(*linear_param.range, n_lin)
        rp, thp = np.meshgrid(rp, thp)
        cp = colors
        type = 'door'
    cbar = ax.pcolormesh(thp, rp, cp, vmin=0, vmax=vmax, cmap='viridis')
    ax.tick_params(axis='x', colors='white')
    #ax.set_yticklabels([])

    if not points is None:
        for (x, c) in points:
            point = get_point_from_x(x, linear_param, angular_param, type)
            ax.scatter(*point, c=c, s=3)

    return cbar

def add_colorbar(fig, im):
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

def create_gpucb_dataset(n_interactions, n_bbs, args):
    """
    :param L: The number of BusyBoxes to include in the dataset.
    :param M: The number of interactions per BusyBox.
    :return:
    """
    # Create a dataset of L busyboxes.
    if args.bb_fname == '':
        bb_dataset_args = Namespace(max_mech=1,
                                    mech_types=args.mech_types,
                                    urdf_num=args.urdf_num,
                                    debug=False,
                                    n_bbs=n_bbs,
                                    n_samples=1,
                                    viz=False,
                                    random_policies=False,
                                    randomness=1.0,
                                    goal_config=None,
                                    bb_fname=None,
                                    no_gripper=True)
        busybox_data = generate_dataset(bb_dataset_args, None)
        print('BusyBoxes created.')
    else:
        # Load in a file with predetermined BusyBoxes.
        with open(args.bb_fname, 'rb') as handle:
            busybox_data = pickle.load(handle)
        busybox_data = [bb_results[0] for bb_results in busybox_data][:n_bbs]

    '''
    # Do a GP-UCB interaction and return Result tuples.
    if os.path.exists(args.fname):
        with open(args.fname, 'rb') as handle:
            dataset = pickle.load(handle)
        n_collected = len(dataset)//n_interactions
        busybox_data = busybox_data[n_collected:]
        print('Already Collected: %d\tRemaining: %d' % (n_collected, len(busybox_data)))
    else:
        dataset = []
    '''
    dataset = []
    regrets = []
    print(len(busybox_data), len(busybox_data[0]))
    for ix, bb_results in enumerate(busybox_data):
        bb_result = bb_results[0]
        single_dataset, _, r = create_single_bb_gpucb_dataset(bb_result,
                                n_interactions,
                                '',
                                args.plot,
                                args,
                                ix,
                                ret_regret=True)
        dataset.extend(single_dataset)
        regrets.append(r)
        print('Interacted with BusyBox %d.' % ix)

    print('Regret:', np.mean(regrets))

    # Save the dataset.
    if args.fname != '':
        with open(args.fname, 'wb') as handle:
            pickle.dump(dataset, handle)

def create_single_bb_gpucb_dataset(bb_result, n_interactions, nn_fname, plot, args, bb_i,
                                    plot_dir_prefix='', ret_regret=False):
    dataset = []

    # interact with BB
    bb = BusyBox.bb_from_result(bb_result, urdf_num=args.urdf_num)
    mech = bb._mechanisms[0]
    no_gripper = True
    image_data, gripper = setup_env(bb, False, False, no_gripper)
    pose_handle_base_world = mech.get_pose_handle_base_world()
    sampler = UCB_Interaction(bb, image_data, plot, args, nn_fname=nn_fname)
    for ix in range(n_interactions):
        # sample a policy
        policy, q = sampler.sample()

        # execute
        traj = policy.generate_trajectory(pose_handle_base_world, q, debug=False)
        c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False)
        gripper.reset(mech)

        result = util.Result(policy.get_policy_tuple(), mech.get_mechanism_tuple(), \
                             motion, c_motion, handle_pose_final, handle_pose_final, \
                             q, image_data, None, 1.0, no_gripper)
        dataset.append(result)

        # update GP
        sampler.update(result)
        if ix % 5 == 0 and args.debug:
            viz_plots(sampler.xs, sampler.ys, sampler.gp)

    opt_points = []
    sample_points = []
    if ret_regret:
        regret, start_x, stop_x = test_model(sampler, args)
        opt_points = [(start_x, 'g'), (stop_x, 'r')]

    if plot:
        policy_plot_data = Policy.get_plot_data(mech)
        sample_points = [(sample, 'r') for sample in sampler.xs]
        viz_circles(image_data, mech, sample_points=sample_points, opt_points=opt_points,
                    gp=sampler.gp, nn=sampler.nn, \
                    bb_i=bb_i, plot_dir_prefix=plot_dir_prefix)
        # viz_plots(sampler.xs, sampler.ys, sampler.gp)

    if ret_regret:
        return dataset, sampler.gp, regret
    else:
        return dataset, sampler.gp

def viz_plots(xs, ys, gp):
    print(xs)
    print(ys)
    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    axes = axes.flatten()
    # For the first plot, plot the policies we have tried.
    for x in xs:
        axes[0].scatter(x[0], x[3])

    # Bin the roll parameter.
    rolls = np.linspace(0, 2*np.pi, num=35)
    for ix, r in enumerate(rolls):
        new_xs = []
        qs = np.linspace(-np.pi/2.0, 0, num=100)
        for q in qs:
            new_xs.append([r, xs[0][1], xs[0][2], q])
        ys, std = gp.predict(new_xs, return_std=True)
        ys = ys.flatten()
        axes[ix+1].plot(qs, ys)
        axes[ix + 1].plot(qs, ys+std, c='r')
        axes[ix + 1].plot(qs, ys + np.sqrt(BETA)*std, c='g')
        axes[ix + 1].plot(qs, ys-std, c='r')
        axes[ix+1].set_title('roll=%.2f' % np.rad2deg(r))
        axes[ix+1].set_ylim(0, 0.2)
    axes[0].set_ylabel('q')
    axes[0].set_xlabel('roll')
    plt.show()

# this executive is for generating GP-UCB interactions from no search_parent_directories
# typically used to generate datasets for training, but can also be used in the L=0
# evaluation case
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-gp-samples',
        type=int,
        default=500,
        help='number of samples to use when fitting a GP to data')
    parser.add_argument(
        '--M',
        type=int,
        help='number of interactions within a single BusyBox during training time')
    parser.add_argument(
        '--L',
        type=int,
        help='number of BusyBoxes to interact with during training time')
    parser.add_argument(
        '--urdf-num',
        default=0,
        help='number to append to generated urdf files. Use if generating multiple datasets simultaneously.')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--mech-types',
        nargs='+',
        default='slider',
        type=str,
        help='if no bb-fname is specified, list the mech types desired')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='use to generate polar plots durin GP-UCB interactions')
    parser.add_argument(
        '--fname',
        default='',
        help='path to save resulting dataset to')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    args = parser.parse_args()
    print(args)
    if args.debug:
        import pdb; pdb.set_trace()

    create_gpucb_dataset(args.M, args.L, args)
