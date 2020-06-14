from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
from collections import OrderedDict
import pickle
import os
import sys
import argparse
from argparse import Namespace
from scipy.optimize import minimize
import itertools
from utils import util
from utils.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
import operator
import torch
from learning.dataloaders import PolicyDataset, parse_pickle_file
from gen.generate_policy_data import get_bb_dataset
from actions.policies import Policy, generate_policy, Revolute, Prismatic, \
                                    get_policy_from_tuple, get_policy_from_x
from learning.gp.viz_doors import viz_3d_plots
from learning.gp.viz_polar_plots import viz_circles
import time

BETA = 2

# takes in a policy and returns and optimization x and the variable bounds
def get_x_and_bounds_from_tuple(policy_params):
    param_vals = policy_params.params
    param_data = policy_params.param_data
    x, bounds = [], []
    for param_name, param_val in param_vals.items():
        if param_name in param_data:
            if param_data[param_name].varied:
                x.append(param_val)
                bounds.append(param_data[param_name].bounds)
    return x, bounds

def process_data(data, n_train):
    """
    Takes in a dataset in our typical format and outputs the dataset to fit the GP.
    :param data:
    :param n_train:
    :return:
    """
    xs, ys = [], []
    for entry in data[0:n_train]:
        x, _ = get_x_and_bounds_from_tuple(entry.policy_params)
        xs.append(x)
        ys.append(entry.net_motion)

    X = np.array(xs)
    Y = np.array(ys).reshape(-1, 1)
    return X, Y


def get_nn_preds(results, model, ret_dataset=False, use_cuda=False):
    data = parse_pickle_file(results)
    dataset = PolicyDataset(data)
    pred_motions = []
    for i in range(len(dataset.items)):
        policy_type = dataset.items[i]['type']
        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
        policy_tensor = dataset.tensors[i].unsqueeze(0)
        image_tensor = dataset.images[i].unsqueeze(0)
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            image_tensor = image_tensor.cuda()
        pred_motion, _ = model.forward(policy_type_tensor,
                                       policy_tensor,
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

def test_model(sampler, args, gripper=None):
    """
    Maximize the GP mean function to get the best policy.
    :param sampler: A GP fit to the current BusyBox.
    :return: Regret.
    """
    # Optimize the GP to get the best policy.
    ucb = False
    stop_x, stop_policy, start_x = sampler.optim.optimize_gp(ucb)

    # Execute the policy and observe the true motion.
    debug = False
    viz = False
    use_gripper = False
    if gripper is None:
        _, gripper = setup_env(sampler.bb, viz, debug, use_gripper)
    else:
        gripper.reset(sampler.mech)
    pose_handle_base_world = sampler.mech.get_pose_handle_base_world()
    traj = stop_policy.generate_trajectory(pose_handle_base_world, debug=debug)
    cmotion, motion, _ = gripper.execute_trajectory(traj, sampler.mech, stop_policy.type, debug=debug)

    # Calculate the regret.
    max_d = sampler.mech.get_max_net_motion()
    regret = (max_d - motion)/max_d

    return regret, start_x, stop_x, stop_policy.type


class GPOptimizer(object):

    def __init__(self, urdf_num, bb, image_data, n_samples, beta, gps, random_policies, nn=None):
        """
        Initialize one of these for each BusyBox.
        """
        self.sample_policies = []
        self.nn_samples = []
        self.nn = nn
        self.mech = bb._mechanisms[0]
        self.beta = beta
        self.gps = gps
        self.n_samples = n_samples
        self.saved_im = None

        # Generate random policies.
        for _ in range(n_samples):
            random_policy = generate_policy(self.mech, random_policies)
            policy_type = random_policy.type
            policy_tuple = random_policy.get_policy_tuple()

            results = [util.Result(policy_tuple, None, 0.0, None, None, None, \
                                    image_data, None, None)]
            self.sample_policies.append(results)

            if self.nn is not None:
                nn_preds, self.dataset = get_nn_preds(results, nn, ret_dataset=True, use_cuda=False)
                self.nn_samples.append(nn_preds)
            else:
                self.dataset = None
                self.nn_samples.append(None)
        # print('Max:', np.max(self.nn_samples))

        self.log = []

    def _optim_result_to_torch(self, policy_type, x, image_tensor, use_cuda=False):
        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
        policy_tensor = torch.tensor(x).float().unsqueeze(0)
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            image_tensor = image_tensor.cuda()

        return [policy_type_tensor, policy_tensor, image_tensor]

    def _objective_func(self, x, policy_type, ucb, image_tensor=None):
        X = np.expand_dims(x, axis=0)

        Y_pred, Y_std = self.gps[policy_type].predict(X, return_std=True)

        if not self.nn is None:
            inputs = self._optim_result_to_torch(policy_type, x, image_tensor, False)
            nn_x = self.nn.policy_modules[policy_type].forward(inputs[1])
            if self.saved_im is None:
                self.saved_im, _ = self.nn.image_module(inputs[2])
            nn_x = torch.cat([nn_x, self.saved_im], dim=1)
            nn_x = F.relu(self.nn.fc1(nn_x))
            nn_x = F.relu(self.nn.fc2(nn_x))
            val = self.nn.fc5(nn_x)

            # val, _ = self.nn.forward(*inputs)
            val = val.detach().numpy()
            Y_pred += val.squeeze()

        if ucb:
            obj = -Y_pred[0] - np.sqrt(self.beta) * Y_std[0]
        else:
            obj = -Y_pred[0]
        return obj

    def _get_pred_motions(self, data, ucb, nn_preds=None):
        X, Y = process_data(data, len(data))
        y_pred, y_std = np.zeros(len(data)), np.zeros(len(data))
        for i, res in enumerate(data):
            y_pred_res, y_std_res = self.gps[res.policy_params.type].predict(X, return_std=True)
            y_pred[i] = y_pred_res
            y_std[i] = y_std_res

        if not nn_preds is None:
            y_pred += np.array(nn_preds)

        if ucb:
            return y_pred + np.sqrt(self.beta) * y_std, self.dataset
        else:
            return y_pred, self.dataset

    def stochastic_gp(self, ucb, temp=0.0075):
        policies = []
        scores = []

        # Generate random policies.
        for res in self.sample_policies:
            # Get predictions from the GP.
            sample_disps, dataset = self._get_pred_motions(res, ucb, nn_preds=None)

            policies.append(res[0].policy_params)
            scores.append(sample_disps[0])

        # Sample a policy based on its score value.
        scores = np.exp(np.array(scores)/temp) #[:, 0]
        scores /= np.sum(scores)

        index = np.random.choice(np.arange(scores.shape[0]),
                                 p=scores)

        self.log.append([scores, index])
        policy_params_max = policies[index]


        # TODO: Make sure the policy appears correctly here.
        x, bounds = get_x_and_bounds_from_tuple(policy_params_max)
        policy = get_policy_from_tuple(policy_params_max)
        return x, policy

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

            samples.append((res[0].policy_params, sample_disps[0]))

        # Find the sample that maximizes the distance.
        policies = sorted(samples, key=operator.itemgetter(1))

        # Start optimization from here.
        if self.nn is None:
            images = None
        else:
            images = dataset.images[0].unsqueeze(0)

        min_val, stop_policy, x_final = float("inf"), None, None
        for policy_params_max, max_disp in policies[-10:]:
            x0, bounds = get_x_and_bounds_from_tuple(policy_params_max)
            opt_res = minimize(fun=self._objective_func, x0=x0,
                                args=(policy_params_max.type, ucb, images),
                                method='L-BFGS-B', options={'eps': 1e-3,
                                                            'maxiter': 1000,
                                                            'gtol': 1e-8,
                                                            'maxls': 50,
                                                            }, bounds=bounds)

            val = opt_res['fun']
            if val <= min_val:
                x_final = opt_res['x']
                stop_policy = get_policy_from_x(self.mech, x_final, policy_params_max)
                min_val = val
        # print(opt_res)
        # print('OPT:', opt_res['success'], opt_res['nit'], opt_res['message'])
        # print('------ Start')
        # print(x0)
        # print(x_final)
        # print(bounds)
        # print('------')
        return x_final, stop_policy, x0


class UCB_Interaction(object):

    def __init__(self, bb, image_data, plot, args, nn_fname='', nn = None):
        # Pretrained Kernel (for Sliders)
        # kernel = ConstantKernel(0.005, constant_value_bounds=(0.005, 0.005)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e2))
        # Pretrained Kernel (for Doors)
        # 0.0202**2 * RBF(length_scale=[0.0533, 0.000248, 0.0327, 0.0278]) + WhiteKernel(noise_level=1e-05)
        self.xs, self.ys, self.moves = {'Prismatic': [], 'Revolute': []}, \
                                        {'Prismatic': [], 'Revolute': []}, \
                                        {'Prismatic': [], 'Revolute': []}

        self.plot = plot
        self.nn = nn
        if nn_fname != '':
            self.nn = util.load_model(nn_fname, args.hdim, use_cuda=False)
        self.nn.eval()
        self.bb = bb
        self.image_data = image_data
        self.mech = self.bb._mechanisms[0]
        self.gps = {'Prismatic': GaussianProcessRegressor(kernel=self.get_kernel('Prismatic', 'random'),
                                               n_restarts_optimizer=1),
                    'Revolute': GaussianProcessRegressor(kernel=self.get_kernel('Revolute', 'random'),
                                                       n_restarts_optimizer=1)}
        self.optim = GPOptimizer(args.urdf_num, self.bb, self.image_data,
                        args.n_gp_samples, BETA, self.gps, args.random_policies, nn=self.nn)

    def get_kernel(self, type, explore_type):
        noise = 1e-5
        variance = 0.005

        if type == 'Prismatic':
            kernel_ls_params = OrderedDict([('pitch', 0.1),
                                ('yaw', 0.1),
                                ('goal_config', 0.1)])
        elif type == 'Revolute':
            if 'random' in explore_type:
                ps = (1.256, 0.018, 0.314)
            else:
                ps = (0.628, 0.009, 0.157)
            # ps = (0.628, 0.009, 0.157)
            print('Using Kernel:', ps)
            kernel_ls_params = OrderedDict([('rot_axis_roll', ps[0]),
                                ('rot_axis_pitch', ps[0]),
                                ('rot_axis_yaw', ps[0]),
                                ('radius_x', ps[1]), # 0.09  # 0.05
                                ('goal_config', ps[2])]) # Keep greater than 0.5
        all_param_data = Policy.get_param_data(type)

        length_scale = []
        length_scale_bounds = []

        for param_name, param_data in all_param_data.items():
            if param_data.varied:
                ls = kernel_ls_params[param_name]
                length_scale.append(ls)
                length_scale_bounds.append((ls, ls))

        return ConstantKernel(variance,
                              constant_value_bounds=(variance, variance)) * \
            RBF(length_scale=length_scale,
                length_scale_bounds=length_scale_bounds) + \
            WhiteKernel(noise_level=noise,
                        noise_level_bounds=(1e-5, 1e2))

    def sample(self, random_policies, stochastic=False):
        # If self.nn is None then make sure each policy type has been
        # attempted at least once
        if self.nn is None and random_policies:
            for policy_class, policy_type in zip([Prismatic, Revolute], \
                                                ['Prismatic', 'Revolute']):
                if len(self.xs[policy_type]) < 1:
                    policy = policy_class._gen(self.mech)
                    policy_tuple = policy.get_policy_tuple()
                    x, _ = get_x_and_bounds_from_tuple(policy_tuple)
                    return x, policy
        # Choose policy using UCB bound.
        ucb = True
        if not stochastic:
            x_final, policy_final, _ = self.optim.optimize_gp(ucb)
        else:
            x_final, policy_final = self.optim.stochastic_gp(ucb)
        return x_final, policy_final

    def update(self, result, x):
        # TODO: Update without the NN.

        # Update GP.
        policy_type = result.policy_params.type
        self.xs[policy_type].append(x)
        if self.nn is None:
            self.ys[policy_type].append([result.net_motion])
        else:
            inputs = self.optim._optim_result_to_torch(policy_type,
                                                       self.xs[policy_type][-1],
                                                       self.optim.dataset.images[0].unsqueeze(0),
                                                       use_cuda=False)
            nn_pred = self.nn.forward(*inputs)[0]
            nn_pred = nn_pred.detach().numpy().squeeze()
            self.ys[policy_type].append([result.net_motion - nn_pred])

        self.moves[policy_type].append([result.net_motion])
        self.gps[policy_type].fit(np.array(self.xs[policy_type]), np.array(self.ys[policy_type]))
        # Visualize GP.
        # if self.ix % 1 == 0 and self.plot:
        #     params = mech.get_mechanism_tuple().params
        #     print('Range:', params.range/2.0)
        #     print('Angle:', np.arctan2(params.axis[1], params.axis[0]))
        #     print('GP:', gp.kernel_)
        #
        #     #plt.clf()
        #     plt.figure(figsize=(15, 15))
        #     for x, y in zip(xs, ys):
        #         plt.scatter(x[0], x[2], s=200)
        #     plt.title('policy samples')
        #     plt.xlabel('pitch')
        #     plt.ylabel('q')
        #     #plt.savefig('gp_samples_%d.png' % ix)
        #     plt.show()
        #     viz_gp(gp, result, ix, bb, nn=nn)

    def calc_avg_regret(self):
        regrets = []
        max_dist = self.mech.get_max_net_motion()
        for y in self.moves[policy_type]:
            regrets.append((max_dist - y[0])/max_dist)
        if len(regrets) > 0:
            return np.mean(regrets)
        else:
            return 'n/a'



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

def create_gpucb_dataset(n_interactions, n_bbs, args, nn = None):
    """
    :param n_bbs: The number of BusyBoxes to include in the dataset.
    :param n_interactions: The number of interactions per BusyBox.
    :param args:
    :return:
    """
    busybox_data = get_bb_dataset(args.bb_fname, n_bbs, args.mech_types, 1, args.urdf_num)

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
    #regrets = []
    for ix, bb_results in enumerate(busybox_data):
        single_dataset, _ = create_single_bb_gpucb_dataset(bb_results[0],
                                                              '',
                                                              args.plot,
                                                              args,
                                                              ix,
                                                              n_interactions=n_interactions,
                                                              plot_dir_prefix=args.plot_dir, nn=nn)
        dataset.append(single_dataset)
        print('Interacted with BusyBox %d.' % ix)
    return dataset
    # Save the dataset.
    # if args.fname != '':
    #     with open(args.fname, 'wb') as handle:
    #         pickle.dump(dataset, handle)

def create_single_bb_gpucb_dataset(bb_result, nn_fname, plot, args, bb_i,
                                   n_interactions=None, plot_dir_prefix='',
                                   ret_regret=False, success_regret=None, nn=None):
    use_cuda = False
    dataset = []
    viz = False
    debug = False
    use_gripper = False
    # interact with BB
    bb = BusyBox.bb_from_result(bb_result, urdf_num=args.urdf_num)
    mech = bb._mechanisms[0]
    image_data, gripper = setup_env(bb, viz, debug, use_gripper)

    pose_handle_base_world = mech.get_pose_handle_base_world()
    sampler = UCB_Interaction(bb, image_data, plot, args, nn_fname=nn_fname, nn=nn)
    for ix in itertools.count():
        gripper.reset(mech)
        if args.debug:
            sys.stdout.write('\rProcessing sample %i' % ix)
        # if we are done sampling n_interactions OR we need to get regret after
        # each interaction
        if ((not n_interactions is None) and ix==n_interactions) or \
            (not success_regret is None):

            regret, start_x, stop_x, policy_type = test_model(sampler, args, gripper=gripper)
            gripper.reset(mech)

            # print('Current regret', regret)
            opt_points = (policy_type, [(start_x, 'g'), (stop_x, 'r')])
            sample_points = {'Prismatic': [(sample, 'k') for sample in sampler.xs['Prismatic']],
                            'Revolute': [(sample, 'k') for sample in sampler.xs['Revolute']]}

            # if done sampling n_interactions
            if (not n_interactions is None) and ix==n_interactions:
                if plot:
                    viz_circles(util.GP_PLOT,
                                image_data,
                                mech,
                                BETA,
                                sample_points=sample_points,
                                opt_points=opt_points,
                                gps=sampler.gps,
                                nn=sampler.nn,
                                bb_i=bb_i,
                                plot_dir_prefix=plot_dir_prefix)
                    plt.show()
                    input('Enter to close')
                    plt.close()
                if ret_regret:
                    return dataset, sampler.gps, regret
                else:
                    return dataset, sampler.gps
            # if got successful interaction or timeout
            elif (not success_regret is None) and \
                            ((regret < success_regret) or (ix >= 100)):
                if plot:
                    viz_circles(util.GP_PLOT,
                                image_data,
                                mech,
                                BETA,
                                sample_points=sample_points,
                                opt_points=opt_points,
                                gps=sampler.gps,
                                nn=sampler.nn,
                                bb_i=bb_i,
                                plot_dir_prefix=plot_dir_prefix)
                return dataset, sampler.gps, ix

        # sample a policy
        # image_data, gripper = setup_env(bb, False, debug, use_gripper)

        gripper.reset(mech)
        x, policy = sampler.sample(args.random_policies, stochastic=args.stochastic)

        # execute
        traj = policy.generate_trajectory(pose_handle_base_world, debug=debug)
        c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False)
        result = util.Result(policy.get_policy_tuple(), mech.get_mechanism_tuple(),
                             motion, c_motion, handle_pose_final, handle_pose_final,
                             image_data, None, True)
        dataset.append(result)

        # update GP
        sampler.update(result, x)
        # uncomment to generate a plot after each sample (WARNING: VERY SLOW!)
        '''
        if plot:
            sample_points = {'Prismatic': [(sample, 'k') for sample in sampler.xs['Prismatic']],
                            'Revolute': [(sample, 'k') for sample in sampler.xs['Revolute']]}
            viz_circles(util.GP_PLOT,
                        image_data,
                        mech,
                        BETA,
                        sample_points=sample_points,
                        opt_points=[],
                        gps=sampler.gps,
                        nn=sampler.nn,
                        bb_i=bb_i,
                        plot_dir_prefix=plot_dir_prefix)
            plt.show()
            input('Enter to close')
            plt.close()

        '''

        # this code has not been updated since refactor
        if ix % 10 == 0 and False:
            if len(nn_fname) > 0:
                model = util.load_model(nn_fname, args.hdim, use_cuda=use_cuda)

            def _gp_callback(new_xs, bb_result):
                ys, std = sampler.gps['Revolute'].predict(new_xs, return_std=True)
                ys = ys.flatten()

                if len(nn_fname) > 0:
                    data = []
                    for pitch, radius, q in new_xs:
                        data.append({
                            'type': 'Revolute',
                            'params': [pitch, radius, q],
                            'goal_config': q,
                            'image': bb_result.image_data,
                            'y': 0.,
                            # 'mech': mech_params,
                        })

                    dataset = PolicyDataset(data)
                    nn_ys = []
                    for i in range(len(dataset.items)):
                        policy_type = dataset.items[i]['type']
                        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
                        policy_tensor = dataset.tensors[i].unsqueeze(0)
                        # config_tensor = dataset.configs[i].unsqueeze(0)
                        image_tensor = dataset.images[i].unsqueeze(0)
                        if use_cuda:
                            policy_type_tensor = policy_type_tensor.cuda()
                            policy_tensor = policy_tensor.cuda()
                            # config_tensor = config_tensor.cuda()
                            image_tensor = image_tensor.cuda()
                        pred_motion, _ = model.forward(policy_type_tensor,
                                                       policy_tensor,
                                                       # config_tensor,
                                                       image_tensor)
                        if True:
                            pred_motion_float = pred_motion.cpu().detach().numpy()[0][0]
                        else:
                            pred_motion_float = pred_motion.detach().numpy()[0][0]
                        nn_ys += [pred_motion_float]
                    ys += np.array(nn_ys)
                return ys, std, ys + np.sqrt(BETA)*std

            viz_3d_plots(xs=sampler.xs[policy_type],
                         callback=_gp_callback,
                         bb_result=bb_result)
        #     viz_radius_plots(sampler.xs, sampler.gp)
        #     viz_plots(sampler.xs, sampler.gp)

    #with open('log_%d.pickle' % bb_i, 'wb') as handle:
    #    pickle.dump(sampler.optim.log, handle)


def viz_radius_plots(xs, gp):
    fig, axes = plt.subplots(6, 6, figsize=(22, 22))
    axes = axes.flatten()
    # For the first plot, plot the policies we have tried.
    for x in xs:
        axes[0].scatter(x[2], x[3])

    # Bin the roll parameter.
    radii = np.linspace(0.08, 0.15, num=35)
    for ix, r in enumerate(radii):
        new_xs = []
        qs = np.linspace(-np.pi/2.0, 0, num=100)
        for q in qs:
            new_xs.append([0, xs[0][1], r, q])
        ys, std = gp.predict(new_xs, return_std=True)
        ys = ys.flatten()
        axes[ix+1].plot(qs, ys)
        axes[ix + 1].plot(qs, ys+std, c='r')
        axes[ix + 1].plot(qs, ys + np.sqrt(BETA)*std, c='g')
        axes[ix + 1].plot(qs, ys-std, c='r')
        axes[ix+1].set_title('radius=%.2f' % r)
        axes[ix+1].set_ylim(0, 0.2)
    axes[0].set_ylabel('q')
    axes[0].set_xlabel('roll')
    plt.show()


def viz_plots(xs, gp):
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
        default=500,# 500,
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
        default=['slider'],
        type=str,
        help='if no bb-fname is specified, list the mech types desired')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='use to generate polar plots durin GP-UCB interactions')
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='')
    parser.add_argument(
        '--fname',
        default='',
        help='path to save resulting dataset to')
    parser.add_argument(
        '--nn-fname',
        default='',
        help='path to save resulting dataset to')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='use to sample from the acquistion function instead of optimizing')
    parser.add_argument(
        '--random-policies',
        action='store_true',
        help='use to try random policy classes on random mechanisms')
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    try:
        create_gpucb_dataset(args.M, args.L, args)
    except:
        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
