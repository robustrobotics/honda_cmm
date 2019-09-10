from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum, ConstantKernel, ExpSineSquared
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from scipy.optimize import minimize
from util import util
from util.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
from actions import policies
import operator
import pybullet as p
from actions.gripper import Gripper
from learning.test_model import get_pred_motions as get_nn_preds
import torch
from learning.dataloaders import PolicyDataset, parse_pickle_file
from gen.generate_policy_data import generate_dataset
from collections import namedtuple

def process_data(data, n_train):
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

    x_preds = []
    for theta in np.linspace(-np.pi, 0, num=100):
        x_preds.append([theta, 0, entry.mechanism_params.params.range/2.0])
    X_pred = np.array(x_preds)

    return X, Y, X_pred


def optim_result_to_torch(x, image_tensor, use_cuda=False):
    policy_type_tensor = torch.Tensor([util.name_lookup['Prismatic']])
    policy_tensor = torch.tensor([[x[0], 0.0]]).float()  # hard code yaw to be 0
    config_tensor = torch.tensor([[x[-1]]]).float()
    if use_cuda:
        policy_type_tensor = policy_type_tensor.cuda()
        policy_tensor = policy_tensor.cuda()
        config_tensor = config_tensor.cuda()
        image_tensor = image_tensor.cuda()

    return [policy_type_tensor, policy_tensor, config_tensor, image_tensor]


def objective_func(x, gp, ucb, beta=100, nn=None, image_tensor=None, use_cuda=False):
    x = x.squeeze()

    X = np.array([[x[0], 0.0, x[-1]]])
    Y_pred, Y_std = gp.predict(X, return_std=True)
    if not nn is None:
        inputs = optim_result_to_torch(x, image_tensor, use_cuda)
        val = nn.forward(*inputs)
        if use_cuda:
            val = val.cpu()
        val = val.detach().numpy()
        Y_pred += val.squeeze()

    if ucb:
        obj = -Y_pred[0] - np.sqrt(beta)*Y_std[0]
    else:
        obj = -Y_pred[0]

    return obj


def get_pred_motions(data, model, ucb, beta=100, nn=None):
    X, Y, _ = process_data(data, len(data))
    y_pred, y_std = model.predict(X, return_std=True)
    dataset = None
    if not nn is None:
        nn_preds, dataset = get_nn_preds(data, nn, ret_dataset=True)
        y_pred += np.array(nn_preds)

    if ucb:
        return y_pred + np.sqrt(beta)*y_std, dataset
    else:
        return y_pred, dataset


def optimize_gp(gp, result, ucb=False, beta=100, nn=None):
    """

    :param gp:
    :param result:
    :param ucb: If True, optimize the UCB objective instead of just the GP.
    :return:
    """
    n_samples = 1000
    samples = []
    bb = BusyBox.bb_from_result(result)
    image_data = setup_env(bb, viz=False, debug=False)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()
    # Generate random policies.
    for _ in range(n_samples):
        # Get the mechanism from the dataset.
        random_policy = policies.generate_policy(bb, mech, True, 1.0)
        policy_type = random_policy.type
        q = random_policy.generate_config(mech, None)
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                None, None, q, image_data, None, 1.0)]
        # Get predictions from the GP.
        sample_disps, dataset = get_pred_motions(results, gp, ucb, beta, nn)


        samples.append(((policy_type,
                        policy_tuple,
                        q,
                        policy_tuple.delta_values.delta_yaw,
                        policy_tuple.delta_values.delta_pitch),
                        sample_disps[0]))

    # Find the sample that maximizes the distance.
    (policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp = max(samples, key=operator.itemgetter(1))

    # Start optimization from here.
    x0 = np.concatenate([[params_max.params.pitch], [q_max]]) # only searching space of pitches!
    if nn is None:
        images = None
    else:
        images = dataset.images[0].unsqueeze(0)
    res = minimize(fun=objective_func, x0=x0, args=(gp, ucb, beta, nn, images, False),
                   method='L-BFGS-B', options={'eps': 10**-3}, bounds=[(-np.pi, 0), (-0.25, 0.25)])
    x_final = res['x']

    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+[x_final[0], 0.0]  # hard code yaw value

    return x_final, dataset


def test_model(gp, result, nn=None):
    """
    Maximize the GP mean function to get the best policy.
    :param gp: A GP fit to the current BusyBox.
    :param result: Result representing the current BusyBox.
    :return: Regret.
    """
    # TODO: Use the NN if required.
    # Optimize the GP to get the best result.
    x_final, _ = optimize_gp(gp, result, ucb=False, nn=nn)

    # Execute the policy and observe the true motion.
    bb = BusyBox.bb_from_result(result)
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


def ucb_interaction(result, max_iterations=50, plot=False, nn_fname='', kx=-1):
    # Pretrained Kernel
    # kernel = ConstantKernel(0.005, constant_value_bounds=(0.005, 0.005)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-5, 1e2))
    variance = 0.005
    noise = 1e-5
    l_pitch, l_yaw, l_q = 0.247, 100, 0.07# #0.0592

    kernel = ConstantKernel(variance, constant_value_bounds=(variance, variance)) * RBF(length_scale=(l_pitch, l_yaw, l_q), length_scale_bounds=((l_pitch, l_pitch), (l_yaw, l_yaw), (l_q, l_q))) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-5, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10)

    # Load the NN if required.
    nn = None
    if nn_fname != '':
        nn = util.load_model(nn_fname, 16)

    interaction_data = []
    xs, ys, moves = [], [], []
    for ix in range(0, max_iterations):
        # (1) Choose a point to interact with.
        if len(xs) < 1 and (nn is None):
            # (a) Choose policy randomly.
            bb = BusyBox.bb_from_result(result)
            image_data = setup_env(bb, viz=False, debug=False)
            mech = bb._mechanisms[0]
            pose_handle_base_world = mech.get_pose_handle_base_world()
            policy = policies.generate_policy(bb, mech, True, 1.0)
            q = policy.generate_config(mech, None)
        else:
            # (b) Choose policy using UCB bound.
            x_final, dataset = optimize_gp(gp, result, ucb=True, beta=40, nn=nn)

            bb = BusyBox.bb_from_result(result)
            image_data = setup_env(bb, viz=False, debug=False)
            mech = bb._mechanisms[0]
            pose_handle_base_world = mech.get_pose_handle_base_world()
            policy_list = list(pose_handle_base_world.p) + [0., 0., 0., 1.] + [x_final[0], 0.0]
            policy = policies.get_policy_from_params('Prismatic', policy_list, mech)
            q = x_final[-1]

        # (2) Interact with BusyBox to get result.
        traj = policy.generate_trajectory(pose_handle_base_world, q, True)
        gripper = Gripper(bb.bb_id)
        c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False)
        p.disconnect()

        result = util.Result(policy.get_policy_tuple(), mech.get_mechanism_tuple(), \
                             motion, c_motion, handle_pose_final, handle_pose_final, \
                             q, image_data, None, -1)
        interaction_data.append(result)

        if ix == 0:
            print('GP:', gp.kernel)
            viz_gp_circles(gp, result, ix, bb, image_data, nn=nn, kx=kx)

        # (3) Update GP.
        policy_params = policy.get_policy_tuple()
        xs.append([policy_params.params.pitch,
                   policy_params.params.yaw,
                   q])
        if nn is None:
            ys.append([motion])
        else:
            inputs = optim_result_to_torch(x_final, dataset.images[0].unsqueeze(0), False)
            nn_pred = nn.forward(*inputs).detach().numpy().squeeze()
            ys.append([motion-nn_pred])
        moves.append([motion])
        gp.fit(np.array(xs), np.array(ys))

        # (4) Visualize GP.
        if ix % 10 == 0 and plot:
            params = mech.get_mechanism_tuple().params
            print('Range:', params.range/2.0)
            print('Angle:', np.arctan2(params.axis[1], params.axis[0]))
            print('GP:', gp.kernel_)

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
            viz_gp_circles(gp, result, ix, bb, image_data, nn=nn, kx=kx, points=xs)

    regrets = []
    for y in moves:
        max_d = mech.get_mechanism_tuple().params.range/2.0
        regrets.append((max_d - y[0])/max_d)
        print(y, max_d)
    return gp, nn, np.mean(regrets), interaction_data


def format_batch(X_pred, bb):
    data = []
    image_data = setup_env(bb, viz=False, debug=False)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()

    for ix in range(X_pred.shape[0]):
        pose_handle_base_world = mech.get_pose_handle_base_world()
        policy_list = list(pose_handle_base_world.p) + [0., 0., 0., 1.] + [X_pred[ix, 0], 0.0]
        policy = policies.get_policy_from_params('Prismatic', policy_list, mech)

        result = util.Result(policy.get_policy_tuple(), mech_tuple, 0.0, 0.0,
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


def viz_gp_circles(gp, result, num, bb, image_data, nn=None, kx=-1, points=None):
    n_pitch = 80
    n_q = 40

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
            loader = format_batch(X_pred, bb)
            k, x, q, im, _, _ = next(iter(loader))
            pol = torch.Tensor([util.name_lookup[k[0]]])
            nn_preds = nn(pol, x, q, im).detach().numpy()
            Y_pred += nn_preds

        for jx in range(0, n_q):
            mean_colors[ix, jx] = Y_pred[jx, 0]
            std_colors[ix, jx] = Y_std[jx]

    plt.clf()
    plt.figure(figsize=(20, 5))
    ax0 = plt.subplot(131)
    w, h, im = image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax0.imshow(np_im)
    mps = bb._mechanisms[0].get_mechanism_tuple().params
    print('Angle:', np.rad2deg(np.arctan2(mps.axis[1], mps.axis[0])))
    ax1 = plt.subplot(132, projection='polar')
    ax1.set_title('mean')
    max_d = bb._mechanisms[0].get_mechanism_tuple().params.range / 2.0
    polar_plots(ax1, mean_colors, vmax=max_d)

    ax2 = plt.subplot(133, projection='polar')
    ax2.set_title('variance')
    polar_plots(ax2, std_colors, vmax=None, points=points)
    # plt.show()
    plt.savefig('gp_polar_bb_%d_%d.png' % (kx, num), bbox_inches='tight')


def polar_plots(ax, colors, vmax, points=None):
    n_pitch, n_q = colors.shape
    thp, rp = np.linspace(0, 2*np.pi, n_pitch*2), np.linspace(0, 0.25, n_q//2)
    rp, thp = np.meshgrid(rp, thp)
    cp = np.zeros(rp.shape)

    cp[0:n_pitch, :] = colors[:, 0:n_q//2][:, ::-1]
    cp[n_pitch:, :] = np.copy(colors[:, n_q//2:])

    ax.pcolormesh(thp, rp, cp, vmax=vmax)
    if not points is None:
        for x in points:
            print([np.rad2deg(x[0]), x[2]])
            if x[2] < 0:
                ax.scatter(x[0] - np.pi, -1*x[2], c='r', s=10)
            else:
                ax.scatter(x[0], x[2], c='r', s=10)


def fit_random_dataset(data):
    X, Y, X_pred = process_data(data, n_train=args.n_train)

    # kernel = ConstantKernel(1.0) * RBF(length_scale=(1.0, 1.0, 1.0), length_scale_bounds=(1e-5, 1)) + WhiteKernel(noise_level=0.01)
    # kernel = ConstantKernel(1.0) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-3, 1e2)) + WhiteKernel(noise_level=0.01)
    kernel = ConstantKernel(0.00038416, constant_value_bounds=(0.00038416, 0.00038416)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, Y)
    print(gp.kernel_)
    viz_gp(gp, data[0], 1)

    max_x = optimize_gp(gp, data[0])

    Y_pred, Y_std = gp.predict(X_pred, return_std=True)

    plt.plot(X_pred[:, 0], Y_pred, c='r', ls='-')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] + Y_std, c='r', ls='--')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] - Y_std, c='r', ls='--')
    plt.scatter(X[:, 0], Y, c='b')
    max_y = gp.predict(np.array([[max_x[0], 0, max_x[1]]]))
    plt.scatter(max_x[0], max_y[0], c='g')
    plt.show()


def evaluate_k_busyboxes(k, args):
    models = ['',
              # 'sagg_noprior_090919_1506/10bbs.pt',
              # 'sagg_noprior_090919_1506/20bbs.pt',
              # 'sagg_noprior_090919_1506/30bbs.pt',
              # 'sagg_noprior_090919_1506/40bbs.pt',
              # 'sagg_noprior_090919_1506/50bbs.pt',
              '']
    # models = ['sagg_learner100/sagg-learner10.pt',
    #           'sagg_learner100/sagg-learner20.pt',
    #           'sagg_learner100/sagg-learner30.pt',
    #           'sagg_learner100/sagg-learner40.pt',
    #           'sagg_learner100/sagg-learner50.pt',
    #           'sagg_learner100/sagg-learner60.pt',
    #           'sagg_learner100/sagg-learner70.pt',
    #           'sagg_learner100/sagg-learner80.pt',
    #           'sagg_learner100/sagg-learner90.pt',
    #           'sagg_learner100/sagg-learner100.pt',
    #           'data_active_ntrain_50000_epoch_20.pt'
    #           '']#,
              #'data/models/prism_gp_nrun_0_epoch_170.pt',
              #'data/models/prism_gp_25bb_0_epoch_220.pt',
              #'data/models/prism_gp_75bb_nrun_0_epoch_180.pt']
    with open('prism_gp_evals.pickle', 'rb') as handle:
        data = pickle.load(handle)
    results = []
    # k=1
    # data = [data[4]]
    for model in models:
        avg_regrets, final_regrets = [], []
        for ix, result in enumerate(data[:k]):
            print('BusyBox', ix)
            gp, nn, avg_regret, _ = ucb_interaction(result,
                                                    max_iterations=args.n_interactions,
                                                    plot=True,
                                                    nn_fname=model,
                                                    kx=ix)
            regret = test_model(gp, result, nn)
            avg_regrets.append(avg_regret)
            print('AVG:', avg_regret)
            final_regrets.append(regret)
            print('Reg:', regret)
        print('Results')
        print('Avg:', np.mean(avg_regrets))
        print('Final:', np.mean(final_regrets))
        res = {'model': model,
               'avg': np.mean(avg_regrets),
               'final': np.mean(final_regrets)}
        results.append(res)
        # with open('regret_results_10-50_1.pickle', 'wb') as handle:
        #     pickle.dump(results, handle)


def create_gpucb_dataset(L=50, M=200, fname=''):
    """
    :param L: The number of BusyBoxes to include in the dataset.
    :param M: The number of interactions per BusyBox.
    :return:
    """
    # TODO: Load in a file with predetermined BusyBoxes.

    # Create a dataset of L busyboxes.
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

    # TODO: Do a GP-UCB interaction and return Result tuples.
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
    args = parser.parse_args()

    create_gpucb_dataset(L=10, M=200)

    # evaluate_k_busyboxes(10, args)

    # fit_random_dataset(data)



