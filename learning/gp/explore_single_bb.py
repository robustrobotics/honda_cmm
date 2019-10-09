from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum, ConstantKernel, ExpSineSquared
import numpy as np
from gen.generator_busybox import create_simulated_baxter_slider
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from scipy.optimize import minimize
from utils import util
from utils.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
from actions import policies
import operator
import pybullet as p
from actions.gripper import Gripper
from actions.policies import Prismatic
from learning.test_model import get_pred_motions as get_nn_preds
import torch
from learning.dataloaders import PolicyDataset, parse_pickle_file
from gen.generate_policy_data import generate_dataset
from collections import namedtuple
import time

def process_data(data, n_train, true_range):
    """
    Takes in a dataset in our typical format and outputs the dataset to fit the GP.
    :param data:
    :return:
    """
    xs, ys = [], []
    for entry in data[0:n_train]:
        pitch = entry.policy_params.params.pitch
        yaw = entry.policy_params.params.yaw
        q = entry.config_goal
        xs.append([pitch, yaw, q])
        ys.append(entry.net_motion)

    X = np.array(xs)
    Y = np.array(ys).reshape(-1, 1)

    if true_range:
        x_preds = []
        for theta in np.linspace(-np.pi, 0, num=100):
            x_preds.append([theta, 0, true_range])
        X_pred = np.array(x_preds)
        return X, Y, X_pred
    else:
        return X, Y, []

def calc_random_policy(pos, orn):
    pitch = np.random.uniform(-np.pi, 0)
    policy = Prismatic(pos, orn, pitch, 0.0)
    config = np.random.uniform(-.14, .14)
    return policy, config

class GPOptimizer(object):

    def __init__(self, urdf_num, pos, orn, true_range, result=None, nn=None, n_samples=500):
        """
        Initialize one of these for each BusyBox.
        """
        self.sample_policies = []
        self.nn_samples = []
        self.true_range = true_range
        self.nn = nn
        if result:
            self.bb = BusyBox.bb_from_result(result, urdf_num)
        else:
            self.bb = create_simulated_baxter_slider()

        image_data = setup_env(self.bb, viz=False, debug=False)
        mech_tuple = self.bb._mechanisms[0].get_mechanism_tuple()
        p.disconnect()
        self.image_data = image_data

        # Generate random policies.
        for _ in range(n_samples):
            # Get the mechanism from the dataset.
            random_policy, q = calc_random_policy(pos, orn)
            policy_type = random_policy.type
            policy_tuple = random_policy.get_policy_tuple()

            results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                   None, None, q, image_data, None, 1.0)]
            self.sample_policies.append(results)

            if not self.nn is None:
                nn_preds, self.dataset = get_nn_preds(results, nn, ret_dataset=True, use_cuda=False)
                self.nn_samples.append(nn_preds)
            else:
                self.dataset = None
                self.nn_samples.append(None)

    def _optim_result_to_torch(self, x, image_tensor, use_cuda=False):
        policy_type_tensor = torch.Tensor([util.name_lookup['Prismatic']])
        policy_tensor = torch.tensor([[x[0], 0.0]]).float()  # hard code yaw to be 0
        config_tensor = torch.tensor([[x[-1]]]).float()
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            config_tensor = config_tensor.cuda()
            image_tensor = image_tensor.cuda()

        return [policy_type_tensor, policy_tensor, config_tensor, image_tensor]

    def _objective_func(self, x, gp, ucb, beta=100, image_tensor=None):
        x = x.squeeze()

        X = np.array([[x[0], 0.0, x[-1]]])
        Y_pred, Y_std = gp.predict(X, return_std=True)

        if not self.nn is None:
            inputs = self._optim_result_to_torch(x, image_tensor, False)
            val, _ = self.nn.forward(*inputs)
            val = val.detach().numpy()
            Y_pred += val.squeeze()

        if ucb:
            obj = -Y_pred[0] - np.sqrt(beta) * Y_std[0]
        else:
            obj = -Y_pred[0]

        return obj

    def _get_pred_motions(self, data, model, ucb, beta=100, nn_preds=None):
        X, Y, _ = process_data(data, len(data), self.true_range)
        y_pred, y_std = model.predict(X, return_std=True)

        if not nn_preds is None:
            y_pred += np.array(nn_preds)

        if ucb:
            return y_pred + np.sqrt(beta) * y_std, self.dataset
        else:
            return y_pred, self.dataset

    def optimize_gp(self, gp, ucb=False, beta=100):
        """
        Find the input (policy) that maximizes the GP (+ NN) output.
        :param gp: A GP representing the reward function to optimize.
        :param result: util.Result tuple containing the BusyBox object.
        :param ucb: If True, optimize the UCB objective instead of just the GP.
        :param beta: beta paramater for the GP-UCB objective.
        :param nn: If not None, optimize gp(.) + nn(.).
        :return: x_final, the optimal policy according to the current model.
        """
        samples = []

        # Generate random policies.
        for res, nn_preds in zip(self.sample_policies, self.nn_samples):
            # Get predictions from the GP.
            sample_disps, dataset = self._get_pred_motions(res, gp, ucb, beta, nn_preds=nn_preds)

            samples.append((('Prismatic',
                             res[0].policy_params,
                             res[0].config_goal,
                             res[0].policy_params.delta_values.delta_yaw,
                             res[0].policy_params.delta_values.delta_pitch),
                             sample_disps[0]))


        # Find the sample that maximizes the distance.
        (policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp = max(samples, key=operator.itemgetter(1))

        # Start optimization from here.
        if self.nn is None:
            images = None
        else:
            images = dataset.images[0].unsqueeze(0)
        x0 = np.concatenate([[params_max.params.pitch], [q_max]]) # only searching space of pitches!

        res = minimize(fun=self._objective_func, x0=x0, args=(gp, ucb, beta, images),
                       method='L-BFGS-B', options={'eps': 10**-3}, bounds=[(-np.pi, 0), (-0.25, 0.25)])
        x_final = res['x']

        return x_final, dataset


def test_model(gp, result, nn=None, use_cuda=False, urdf_num=0):
    """
    Maximize the GP mean function to get the best policy.
    :param gp: A GP fit to the current BusyBox.
    :param result: Result representing the current BusyBox.
    :return: Regret.
    """
    # Optimize the GP to get the best result.
    optim_gp = GPOptimizer(urdf_num, result, nn=nn, use_cuda=use_cuda, urdf_num=urdf_num)
    x_final, _ = optim_gp.optimize_gp(gp, result, ucb=False, nn=nn, urdf_num=urdf_num)

    # Execute the policy and observe the true motion.
    bb = BusyBox.bb_from_result(result, urdf_num=urdf_num)
    setup_env(bb, viz=False, debug=False)
    mech = bb._mechanisms[0]
    ps = mech.get_mechanism_tuple().params
    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p) + [0., 0., 0., 1.] + [x_final[0], 0.0]
    policy = policies.get_policy_from_params('Prismatic', policy_list, mech)
    q = x_final[-1]

    traj = policy.generate_trajectory(pose_handle_base_world, q, True)
    gripper = Gripper(bb.bb_id)
    _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, False)
    # import time
    # time.sleep(1)
    p.disconnect()

    # Calculate the regret.
    max_d = bb._mechanisms[0].get_mechanism_tuple().params.range/2.0
    regret = (max_d - motion)/max_d

    return regret


class UCB_Interaction(object):

    def __init__(self, urdf_num, result=None, plot=False, true_range=None, pos=None, orn=None, nn_fname=''):
        # Pretrained Kernel
        # kernel = ConstantKernel(0.005, constant_value_bounds=(0.005, 0.005)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e2))
        self.variance = 0.005
        self.noise = 1e-5
        self.l_yaw = 100
        self.l_q = 0.1
        self.l_pitch = 0.10
        self.kernel = ConstantKernel(self.variance, constant_value_bounds=(self.variance, self.variance)) \
                        * RBF(length_scale=(self.l_pitch, self.l_yaw, self.l_q),
                                length_scale_bounds=((self.l_pitch, self.l_pitch),
                                (self.l_yaw, self.l_yaw), (self.l_q, self.l_q))) \
                                + WhiteKernel(noise_level=self.noise, noise_level_bounds=(1e-5, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                      n_restarts_optimizer=10)


        self.interaction_data = []
        self.xs, self.ys, self.moves = [], [], []

        self.ix = 0
        self.plot = plot
        self.nn = None
        if nn_fname != '':
            self.nn = util.load_model(nn_fname, 16, use_cuda=False)

        if result:
            bb = BusyBox.bb_from_result(result)
            setup_env(bb, viz=False, debug=False)
            self.pos = list(bb._mechanisms[0].get_pose_handle_base_world().p)
            self.orn = [0., 0., 0., 1.] # if from result then all policies in this frame
            self.true_range = bb._mechanisms[0].range/2
        else:
            self.pos = list(pos)
            self.orn = list(orn)
            self.true_range = true_range
        self.optim = GPOptimizer(urdf_num, self.pos, self.orn, self.true_range, nn=self.nn, result=result)

    def sample(self):
        # (1) Choose a point to interact with.
        if len(self.xs) < 1 and self.nn is None:
            # (a) Choose policy randomly.
            policy, q = calc_random_policy(self.pos, self.orn)
        else:
            # (b) Choose policy using UCB bound.
            x_final, dataset = self.optim.optimize_gp(self.gp, ucb=True, beta=10)
            policy_list = self.pos + self.orn + [x_final[0], 0.0]
            policy = policies.get_policy_from_params('Prismatic', policy_list)
            q = x_final[-1]
        self.ix += 1
        return policy, q

    def update(self, policy, q, motion):
        # TODO: Update without the NN.

        # (3) Update GP.
        policy_params = policy.get_policy_tuple()
        self.xs.append([policy_params.params.pitch,
                   policy_params.params.yaw,
                   q])
        if self.nn is None:
            self.ys.append([motion])
        else:
            inputs = self.optim._optim_result_to_torch(self.xs[-1], self.optim.dataset.images[0].unsqueeze(0), use_cuda=False)
            nn_pred = self.nn.forward(*inputs)[0]
            nn_pred = nn_pred.detach().numpy().squeeze()
            self.ys.append([motion - nn_pred])

        self.moves.append([motion])
        self.gp.fit(np.array(self.xs), np.array(self.ys))

        # (4) Visualize GP.
        if self.ix % 1 == 0 and self.plot:
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
            viz_gp_circles(self.gp, self.ix, self.true_range, points=self.xs, nn=self.nn, image_data=self.optim.image_data)

    def calc_regrets(self):
        regrets = []
        for y in self.moves:
            regrets.append((self.true_range - y[0])/self.true_range)
            #print(y, true_range)
        return np.mean(regrets)

def format_batch(X_pred, image_data):
    data = []

    for ix in range(X_pred.shape[0]):
        # pose_handle_base_world = mech.get_pose_handle_base_world()

        policy_list = [0., 0., 0.] + [0., 0., 0., 1.] + [X_pred[ix, 0], 0.0]
        policy = policies.get_policy_from_params('Prismatic', policy_list)

        result = util.Result(policy.get_policy_tuple(), None, 0.0, 0.0,
                             None, None, X_pred[ix, -1], image_data, None, 1.0)
        data.append(result)

    data = parse_pickle_file(data=data)
    dataset = PolicyDataset(data)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=len(dataset))
    return train_loader


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
            loader = format_batch(X_pred, bb)
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


def viz_gp_circles(gp, num, max_d, points=[], nn=None, image_data=None):
    n_pitch = 40
    n_q = 20
    bb = create_simulated_baxter_slider()

    radii = np.linspace(-0.25, 0.25, num=n_q)
    thetas = np.linspace(-np.pi, 0, num=n_pitch)
    r, th = np.meshgrid(radii, thetas)
    mean_colors = np.zeros(r.shape)
    std_colors = np.zeros(r.shape)

    for ix in range(0, n_pitch):
        x_preds = []
        for jx in range(0, n_q):
            x_preds.append([thetas[ix], 0, radii[jx]])
        X_pred = np.array(x_preds)

        Y_pred, Y_std = gp.predict(X_pred, return_std=True)
        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        if not nn is None:
            loader = format_batch(X_pred, image_data)
            k, x, q, im, _, _ = next(iter(loader))
            pol = torch.Tensor([util.name_lookup[k[0]]])
            nn_preds = nn(pol, x, q, im)[0].detach().numpy()
            Y_pred += nn_preds

        for jx in range(0, n_q):
            mean_colors[ix, jx] = Y_pred[jx, 0]
            std_colors[ix, jx] = Y_std[jx]

    # plt.clf()
    # f = plt.figure(figsize=(20, 5))
    # f.set_facecolor((0, 0, 0))
    # ax0 = plt.subplot(131)
    # w, h, im = image_data
    # np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    # ax0.imshow(np_im)
    # mps = bb._mechanisms[0].get_mechanism_tuple().params
    # print('Angle:', np.rad2deg(np.arctan2(mps.axis[1], mps.axis[0])))
    # ax1 = plt.subplot(132, projection='polar')
    # ax1.set_title('mean')
    # max_d = bb._mechanisms[0].get_mechanism_tuple().params.range / 2.0
    # polar_plots(ax1, mean_colors, vmax=max_d)
    #
    # ax2 = plt.subplot(133, projection='polar')
    # ax2.set_title('variance')
    # polar_plots(ax2, std_colors, vmax=None, points=points)
    # plt.show()
    #
    # if '/' in model_name:
    #     model_name = model_name.split('/')[-1].replace('.pt', '')
    # fname = 'prior_plots/gp_polar_bb_%d_%d_%s.png' % (kx, num, model_name)
    # plt.savefig(fname, bbox_inches='tight')

    # -------------------------
    plt.clf()
    plt.figure(figsize=(5, 5))
    # ax0 = plt.subplot(111)
    # w, h, im = image_data
    # np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    # ax0.imshow(np_im)

    ax1 = plt.subplot(111, projection='polar')
    max_d = bb._mechanisms[0].get_mechanism_tuple().params.range / 2.0
    polar_plots(ax1, mean_colors, vmax=max_d, points=points)
    ax1.set_title('Reward (T=%d)' % (num + 1), color='w', y=1.15)
    #
    # ax2 = plt.subplot(111, projection='polar')
    # polar_plots(ax2, std_colors, vmax=None, points=points)
    # plt.show()
    if '/' in model_name:
        model_name = model_name.split('/')[-1].replace('.pt', '')
    fname = 'videos/gp_polar_bb_%d_%d_%s_mean_arrow.png' % (kx, num, model_name)
    plt.savefig(fname, bbox_inches='tight', facecolor='k')

    plt.clf()
    plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(111, projection='polar')
    max_d = bb._mechanisms[0].get_mechanism_tuple().params.range / 2.0
    polar_plots(ax1, mean_colors, vmax=max_d, points=None)
    ax1.set_title('Reward (T=%d)' % (num+1), color='w', y=1.15)
    #
    # ax2 = plt.subplot(111, projection='polar')
    # polar_plots(ax2, std_colors, vmax=None, points=points)
    #plt.show()
    if '/' in model_name:
        model_name = model_name.split('/')[-1].replace('.pt', '')
    fname = 'videos/gp_polar_bb_%d_%d_%s_mean.png' % (kx, num, model_name)
    plt.savefig(fname, bbox_inches='tight', facecolor='k')
    # ----------------------------------



def polar_plots(ax, colors, vmax, points=None):
    n_pitch, n_q = colors.shape
    thp, rp = np.linspace(0, 2*np.pi, n_pitch*2), np.linspace(0, 0.25, n_q//2)
    rp, thp = np.meshgrid(rp, thp)
    cp = np.zeros(rp.shape)
    thp += np.pi

    cp[0:n_pitch, :] = colors[:, 0:n_q//2][:, ::-1]
    cp[n_pitch:, :] = np.copy(colors[:, n_q//2:])

    cbar = ax.pcolormesh(thp, rp, cp, vmax=vmax, cmap='viridis')
    ax.tick_params(axis='x', colors='white')
    ax.set_yticklabels([])
    plt.colorbar(cbar, ax=ax, pad=0.2)
    if not points is None:
        for x in points:
            #print([np.rad2deg(x[0]), x[2]])
            if x[2] < 0:
                ax.scatter(x[0] - np.pi, -1*x[2], c='r', s=10)
            else:
                ax.scatter(x[0], x[2], c='r', s=10)


def fit_random_dataset(data):
    X, Y, X_pred = process_data(data, args.n_train, None)

    # kernel = ConstantKernel(1.0) * RBF(length_scale=(1.0, 1.0, 1.0), length_scale_bounds=(1e-5, 1)) + WhiteKernel(noise_level=0.01)
    # kernel = ConstantKernel(1.0) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=0.01)
    kernel = ConstantKernel(0.00038416, constant_value_bounds=(0.00038416, 0.00038416)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, Y)
    print(gp.kernel_)
    viz_gp(gp, data[0], 1)

    max_x = optimize_gp(gp, data[0], urdf_num=urdf_num)

    Y_pred, Y_std = gp.predict(X_pred, return_std=True)

    plt.plot(X_pred[:, 0], Y_pred, c='r', ls='-')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] + Y_std, c='r', ls='--')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] - Y_std, c='r', ls='--')
    plt.scatter(X[:, 0], Y, c='b')
    max_y = gp.predict(np.array([[max_x[0], 0, max_x[1]]]))
    plt.scatter(max_x[0], max_y[0], c='g')
    plt.show()


def evaluate_k_busyboxes(k, args, use_cuda=False):
    '''
    # Active-NN Models
    if args.eval == 'active_nn':
        models = ['conv2_models/model_ntrain_1000.pt',
                  'conv2_models/model_ntrain_2000.pt',
                  'conv2_models/model_ntrain_3000.pt',
                  'conv2_models/model_ntrain_4000.pt',
                  'conv2_models/model_ntrain_5000.pt',
                  'conv2_models/model_ntrain_6000.pt',
                  'conv2_models/model_ntrain_7000.pt',
                  'conv2_models/model_ntrain_8000.pt',
                  'conv2_models/model_ntrain_9000.pt',
                  'conv2_models/model_ntrain_10000.pt']

    # GP-UCB-NN Models
    elif args.eval == 'gpucb_nn':
        models = ['gpucb_data/model_ntrain_1000.pt',
                  'gpucb_data/model_ntrain_2000.pt',
                  'gpucb_data/model_ntrain_3000.pt',
                  'gpucb_data/model_ntrain_4000.pt',
                  'gpucb_data/model_ntrain_5000.pt',
                  'gpucb_data/model_ntrain_6000.pt',
                  'gpucb_data/model_ntrain_7000.pt',
                  'gpucb_data/model_ntrain_8000.pt',
                  'gpucb_data/model_ntrain_9000.pt',
                  'gpucb_data/model_ntrain_10000.pt']

    # Random-NN Models
    elif args.eval == 'random_nn':
        models = ['random_100bb_100int/torch_models/model_ntrain_500.pt',
                    'random_100bb_100int/torch_models/model_ntrain_1000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_1500.pt',
                    'random_100bb_100int/torch_models/model_ntrain_2000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_3000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_4000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_5000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_6000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_7000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_8000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_9000.pt',
                    'random_100bb_100int/torch_models/model_ntrain_1000.pt']

    # GP-UCB Models
    elif args.eval == 'test_good':
        models = ['gpucb_data/model_ntrain_10000.pt']
    elif args.eval == 'test_bad':
        models = ['gpucb_data/model_ntrain_1000.pt']
    else:
        models = ['']
    '''
    with open('prism_gp_evals_square_50.pickle', 'rb') as handle:
        data = pickle.load(handle)

    results = []
    for model in args.models:
        avg_regrets, final_regrets = [], []
        for ix, result in enumerate(data[:k]):
            print('BusyBox', ix)
            gp, nn, avg_regret, interactions = ucb_interaction(result,
                                                    max_iterations=args.n_interactions,
                                                    plot=True,
                                                    nn_fname=model,
                                                    kx=ix,
                                                    use_cuda=use_cuda,
                                                    urdf_num=args.urdf_num)
            create_video(interactions)

            regret = test_model(gp, result, nn, use_cuda=use_cuda, urdf_num=args.urdf_num)
            avg_regrets.append(avg_regret)
            print('AVG:', avg_regret)
            final_regrets.append(regret)
            print('Reg:', regret)
        print('Results')
        print('Avg:', np.mean(avg_regrets))
        print('Final:', np.mean(final_regrets))
        res = {'model': model,
               'avg': np.mean(avg_regrets),
               'final': np.mean(final_regrets),
               'regrets': final_regrets}
        results.append(res)
        print('regret_results_%s_t%d_n%d.pickle' % (args.eval, args.n_interactions, k))
        # with open('regret_results_%s_t%d_n%d.pickle' % (args.eval, args.n_interactions, k), 'wb') as handle:
        #     pickle.dump(results, handle)


def create_gpucb_dataset(L=50, M=200, bb_fname='', fname=''):
    """
    :param L: The number of BusyBoxes to include in the dataset.
    :param M: The number of interactions per BusyBox.
    :return:
    """
    # Create a dataset of L busyboxes.
    if bb_fname == '':
        Args = namedtuple('args', 'max_mech, urdf_num, debug, n_bbs, n_samples, viz, match_policies, randomness, goal_config')
        args = Args(max_mech=1,
                    urdf_num=0,
                    debug=False,
                    n_bbs=L,
                    n_samples=1,
                    viz=False,
                    match_policies=True,
                    randomness=1.0,
                    goal_config=None)
        busybox_data = generate_dataset(args, None)
        print('BusyBoxes created.')
    else:
        # Load in a file with predetermined BusyBoxes.
        with open(bb_fname, 'rb') as handle:
            busybox_data = pickle.load(handle)

    # Do a GP-UCB interaction and return Result tuples.
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            dataset = pickle.load(handle)
        n_collected = len(dataset)//M
        busybox_data = busybox_data[n_collected:]
        print('Already Collected: %d\tRemaining: %d' % (n_collected, len(busybox_data)))
    else:
        dataset = []

    for ix, result in enumerate(busybox_data):
        _, _, _, interaction_data = ucb_interaction(result,
                                                    max_iterations=M,
                                                    plot=False,
                                                    nn_fname='',
                                                    kx=-1)
        dataset.extend(interaction_data)
        print('Interacted with BusyBox %d.' % ix)

        # Save the dataset.
        if fname != '':
            with open(fname, 'wb') as handle:
                pickle.dump(dataset, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int)
    parser.add_argument('--n-interactions', type=int)
    parser.add_argument('--eval', type=str)
    parser.add_argument('--urdf-num', default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--models', nargs='*')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    # create_gpucb_dataset(L=100,
    #                      M=100,
    #                      bb_fname='active_100bbs.pickle',
    #                      fname='gpucb_100bb_100i.pickle')

    # evaluate_k_busyboxes(50, args, use_cuda=True)

    # fit_random_dataset(data)
    results = util.read_from_file('test')
    result=results[0]
    sampler = UCB_Interaction(args.urdf_num, result=result,#pos=(0.,0.,0.),
                                     #orn=(0.,0.,0.,1.),
                                     #true_range=0.3,
                                     plot=True,
                                     nn_fname='../overflow/random_100bb_100int_90/torch_models/model_ntrain_9000.pt')
    policy, q = sampler.sample()
    sampler.update(policy, q, 0.2)
