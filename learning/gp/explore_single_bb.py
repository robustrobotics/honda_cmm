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

    def __init__(self, urdf_num, pos, orn, true_range, image_data, n_samples, nn=None):
        """
        Initialize one of these for each BusyBox.
        """
        self.sample_policies = []
        self.nn_samples = []
        self.true_range = true_range
        self.nn = nn
        self.image_data = image_data

        # Generate random policies.
        for _ in range(n_samples):
            # Get the mechanism from the dataset.
            random_policy, q = calc_random_policy(pos, orn)
            policy_type = random_policy.type
            policy_tuple = random_policy.get_policy_tuple()

            results = [util.Result(policy_tuple, None, 0.0, 0.0,
                                   None, None, q, self.image_data, None, 1.0)]
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


def test_model(gp, bb_result, args, nn=None, use_cuda=False, urdf_num=0):
    """
    Maximize the GP mean function to get the best policy.
    :param gp: A GP fit to the current BusyBox.
    :param result: Result representing the current BusyBox.
    :return: Regret.
    """
    # Optimize the GP to get the best result.
    pos, orn, true_range, image_data, _, _, bb, gripper = get_bb_params(bb_result, args)
    optim_gp = GPOptimizer(urdf_num, pos, orn, true_range, image_data, args.n_gp_samples, nn=nn)
    x_final, _ = optim_gp.optimize_gp(gp, ucb=False)

    # Execute the policy and observe the true motion.
    mech = bb._mechanisms[0]
    ps = mech.get_mechanism_tuple().params
    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p) + [0., 0., 0., 1.] + [x_final[0], 0.0]
    policy = policies.get_policy_from_params('Prismatic', policy_list, mech)
    q = x_final[-1]

    traj = policy.generate_trajectory(pose_handle_base_world, q, True)
    _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, False)
    # import time
    # time.sleep(1)
    p.disconnect()

    # Calculate the regret.
    max_d = bb._mechanisms[0].get_mechanism_tuple().params.range/2.0
    regret = (max_d - motion)/max_d

    return regret


class UCB_Interaction(object):

    def __init__(self, image_data, args, true_range=None, pos=None, orn=None, nn_fname=''):
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
        self.plot = args.plot
        self.nn = None
        if nn_fname != '':
            self.nn = util.load_model(nn_fname, args.hdim, use_cuda=False)

        self.pos = list(pos)
        self.orn = list(orn)
        self.true_range = true_range
        self.optim = GPOptimizer(args.urdf_num, self.pos, self.orn, self.true_range, image_data, args.n_gp_samples, nn=self.nn)

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

    def viz_polar_plots(self):
        viz_gp_circles(self.gp, self.ix, self.true_range, points=self.xs, nn=self.nn, image_data=self.optim.image_data)

    def calc_avg_regret(self):
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
    #plt.clf()
    #plt.figure(figsize=(5, 5))
    ax0 = plt.subplot(121)
    w, h, im = image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax0.imshow(np_im)

    ax1 = plt.subplot(122, projection='polar')
    polar_plots(ax1, mean_colors, vmax=max_d, points=points)
    ax1.set_title('Reward (T=%d)' % (num + 1), color='w', y=1.15)
    #
    # ax2 = plt.subplot(111, projection='polar')
    # polar_plots(ax2, std_colors, vmax=None, points=points)
    # plt.show()
    #if '/' in model_name:
    #    model_name = model_name.split('/')[-1].replace('.pt', '')
    #fname = 'videos/gp_polar_bb_%d_%d_%s_mean_arrow.png' % (kx, num, model_name)
    #plt.savefig(fname, bbox_inches='tight', facecolor='k')

    #plt.clf()
    #plt.figure(figsize=(5, 5))
    #ax1 = plt.subplot(111, projection='polar')
    #max_d = bb._mechanisms[0].get_mechanism_tuple().params.range / 2.0
    #polar_plots(ax1, mean_colors, vmax=max_d, points=None)
    #ax1.set_title('Reward (T=%d)' % (num+1), color='w', y=1.15)
    #
    # ax2 = plt.subplot(111, projection='polar')
    # polar_plots(ax2, std_colors, vmax=None, points=points)
    #plt.show()
    #if '/' in model_name:
    #    model_name = model_name.split('/')[-1].replace('.pt', '')
    #fname = 'videos/gp_polar_bb_%d_%d_%s_mean.png' % (kx, num, model_name)
    #plt.savefig(fname, bbox_inches='tight', facecolor='k')
    # ----------------------------------
    plt.show()


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


def evaluate_models(n_interactions, n_bbs, args, use_cuda=False):
    with open(args.bb_fname, 'rb') as handle:
        data = pickle.load(handle)

    results = []
    for model in args.models:
        avg_regrets, final_regrets = [], []
        for ix, bb_result in enumerate(data[:n_bbs]):
            print('BusyBox', ix)
            dataset, avg_regret, gp = create_single_bb_gpucb_dataset(bb_result, n_interactions, model, args)
            nn = util.load_model(model, args.hdim, use_cuda=False)
            regret = test_model(gp, bb_result, args, nn, use_cuda=use_cuda, urdf_num=args.urdf_num)
            avg_regrets.append(avg_regret)
            print('Average Regret:', avg_regret)
            final_regrets.append(regret)
            print('Test Regret   :', regret)
        print('Results')
        print('Average Regret:', np.mean(avg_regrets))
        print('Final Regret  :', np.mean(final_regrets))
        res = {'model': model,
               'avg': np.mean(avg_regrets),
               'final': np.mean(final_regrets),
               'regrets': final_regrets}
        results.append(res)
        results_fname = 'regret_results_%s_t%d_n%d.pickle'
        print(results_fname % (args.eval, n_interactions, n_bbs))
        with open(results_fname % (args.eval, n_interactions, n_bbs), 'wb') as handle:
            pickle.dump(results, handle)


def create_gpucb_dataset(n_interactions, n_bbs, args):
    """
    :param L: The number of BusyBoxes to include in the dataset.
    :param M: The number of interactions per BusyBox.
    :return:
    """
    # Create a dataset of L busyboxes.
    if args.bb_fname == '':
        bb_dataset_args = Namespace(max_mech=1,
                                    urdf_num=args.urdf_num,
                                    debug=False,
                                    n_bbs=n_bbs,
                                    n_samples=1,
                                    viz=False,
                                    match_policies=True,
                                    randomness=1.0,
                                    goal_config=None,
                                    bb_file=None)
        busybox_data = generate_dataset(bb_dataset_args, None)
        print('BusyBoxes created.')
    else:
        # Load in a file with predetermined BusyBoxes.
        with open(args.bb_fname, 'rb') as handle:
            busybox_data = pickle.load(handle)
        busybox_data = busybox_data[:n_bbs]

    # Do a GP-UCB interaction and return Result tuples.
    if os.path.exists(args.fname):
        with open(args.fname, 'rb') as handle:
            dataset = pickle.load(handle)
        n_collected = len(dataset)//n_interactions
        busybox_data = busybox_data[n_collected:]
        print('Already Collected: %d\tRemaining: %d' % (n_collected, len(busybox_data)))
    else:
        dataset = []

    for ix, bb_result in enumerate(busybox_data):
        single_dataset, _, _ = create_single_bb_gpucb_dataset(bb_result, n_interactions, args.nn_fname, args)
        dataset.extend(single_dataset)
        print('Interacted with BusyBox %d.' % ix)

    # Save the dataset.
    if args.fname != '':
        with open(args.fname, 'wb') as handle:
            pickle.dump(dataset, handle)


def get_bb_params(bb_result, args):
    # initialize BB in pyBullet
    bb = BusyBox.bb_from_result(bb_result, urdf_num=args.urdf_num)
    image_data = setup_env(bb, viz=False, debug=False)
    mech = bb._mechanisms[0]
    pose_handle_base_world = mech.get_pose_handle_base_world()
    pos = pose_handle_base_world.p
    orn = [0., 0., 0., 1.] # if from result then all policies in this frame
    true_range = mech.range/2
    gripper = Gripper()
    return pos, orn, true_range, image_data, mech, pose_handle_base_world, bb, gripper

def create_single_bb_gpucb_dataset(bb_result, n_interactions, nn_fname, args):
    pos, orn, true_range, image_data, mech, pose_handle_base_world, bb, gripper = get_bb_params(bb_result, args)
    dataset = []

    # interact with BB
    sampler = UCB_Interaction(image_data, args, true_range, pos, orn, nn_fname=nn_fname)
    for _ in range(n_interactions):
        # sample a policy
        policy, q = sampler.sample()

        # execute
        traj = policy.generate_trajectory(pose_handle_base_world, q)
        c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False)
        gripper.reset(mech)

        result = util.Result(policy.get_policy_tuple(), mech.get_mechanism_tuple(), \
                             motion, c_motion, handle_pose_final, handle_pose_final, \
                             q, image_data, None, -1)
        dataset.append(result)

        # update GP
        sampler.update(policy, q, motion)

    if args.plot:
        sampler.viz_polar_plots()
    return dataset, sampler.calc_avg_regret(), sampler.gp

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
        '--plot',
        action='store_true',
        help='use to generate polar plots durin GP-UCB interactions')
    parser.add_argument(
        '--nn-fname',
        default='',
        help='path to NN to initialize GP-UCB interactions')
    parser.add_argument(
        '--fname',
        default='',
        help='path to save resulting dataset to')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--hdim',
        type=int,
        default=16,
        help='hdim of supplied model(s), used to load model file'
    )
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    create_gpucb_dataset(args.M, args.L, args)
