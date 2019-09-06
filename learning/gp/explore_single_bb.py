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


def objective_func(x, model, ucb, beta=100):
    x = x.squeeze()

    X = np.array([[x[0], 0.0, x[-1]]])
    Y_pred, Y_std = model.predict(X, return_std=True)

    if ucb:
        obj = -Y_pred[0] - np.sqrt(beta)*Y_std[0]
    else:
        obj = -Y_pred[0]

    return obj


def get_pred_motions(data, model, ucb, beta=100):
    X, Y, _ = process_data(data, len(data))
    if ucb:
        y_pred, y_std = model.predict(X, return_std=True)
        return y_pred + np.sqrt(beta)*y_std
    else:
        return model.predict(X)


def optimize_gp(gp, result, ucb=False, beta=100):
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
        # TODO: Get the mechanism from the dataset.
        random_policy = policies.generate_policy(bb, mech, True, 1.0)
        policy_type = random_policy.type
        q = random_policy.generate_config(mech, None)
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                None, None, q, image_data, None, 1.0)]
        # TODO: Get predictions from the GP.
        sample_disps = get_pred_motions(results, gp, ucb, beta)


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
    res = minimize(fun=objective_func, x0=x0, args=(gp, ucb, beta),
                   method='L-BFGS-B', options={'eps': 10**-3}, bounds=[(-np.pi, 0), (-0.1, 0.1)])
    x_final = res['x']

    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+[x_final[0], 0.0]  # hard code yaw value

    return x_final


def ucb_interaction(result, max_iterations=50):
    # Create the BusyBox.
    bb = BusyBox.bb_from_result(result)
    image_data = setup_env(bb, viz=False, debug=False)
    mech_params = bb._mechanisms[0].get_mechanism_tuple()
    print(mech_params)
    print(np.arctan2(mech_params.params.axis[1], mech_params.params.axis[0]))

    # Create a GP.
    # 0.0328**2 * RBF(length_scale=0.0705) + WhiteKernel(noise_level=1e-05)
    kernel = ConstantKernel(1) * RBF(length_scale=1, length_scale_bounds=(1e-1, 1e2)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e5))
    kernel = ConstantKernel(0.04, constant_value_bounds=(0.02, 0.04)) * RBF(length_scale=(0.05, 0.05, 0.05), length_scale_bounds=(1e-2, 1)) + WhiteKernel(noise_level=0.01,
                                                                                                    noise_level_bounds=(
                                                                                                    1e-5, 1e5))
    kernel = ConstantKernel(0.00038416, constant_value_bounds=(0.00038416, 0.00038416)) * RBF(length_scale=(0.247, 0.084, 0.0592), length_scale_bounds=(0.0592, 0.247)) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=10)

    xs, ys = [], []
    for ix in range(0, max_iterations):
        # (1) Choose a point to interact with.
        if len(xs) < 10:
            # (a) Choose policy randomly.
            setup_env(bb, viz=False, debug=False)
            mech = bb._mechanisms[0]
            pose_handle_base_world = mech.get_pose_handle_base_world()
            policy = policies.generate_policy(bb, mech, True, 1.0)
            q = policy.generate_config(mech, None)
        else:
            # (b) Choose policy using UCB bound.
            x_final = optimize_gp(gp, result, ucb=True, beta=40)

            setup_env(bb, viz=False, debug=False)
            mech = bb._mechanisms[0]
            pose_handle_base_world = mech.get_pose_handle_base_world()
            policy_list = list(pose_handle_base_world.p) + [0., 0., 0., 1.] + [x_final[0], 0.0]
            policy = policies.get_policy_from_params('Prismatic', policy_list, mech)
            q = x_final[-1]

        # (2) Interact with BusyBox to get result.
        traj = policy.generate_trajectory(pose_handle_base_world, q, True)
        gripper = Gripper(bb.bb_id)
        _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, False)
        p.disconnect()

        # (3) Update GP.
        policy_params = policy.get_policy_tuple()
        xs.append([policy_params.params.pitch,
                   policy_params.params.yaw,
                   q])
        ys.append([motion])
        gp.fit(np.array(xs), np.array(ys))
        print(gp.kernel_)

        # (4) Visualize GP.
        print(xs[-1])
        if ix % 10 == 0:

            #plt.clf()
            plt.figure(figsize=(15, 15))
            for x, y in zip(xs, ys):
                plt.scatter(x[0], x[2], s=200)
            plt.title('policy samples')
            plt.xlabel('pitch')
            plt.ylabel('q')
            #plt.savefig('gp_samples_%d.png' % ix)
            plt.show()

            viz_gp(gp, result, ix)

def viz_gp(gp, result, num):
    n_pitch = 10
    #plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=10, sharey=True, sharex=True, figsize=(40, 4))
    axes = axes.flatten()
    for ix, pitch in enumerate(np.linspace(-np.pi, 0, n_pitch)):
        x_preds = []
        for q in np.linspace(-0.1, 0.1, num=100):
            x_preds.append([pitch, 0, q])
        X_pred = np.array(x_preds)

        Y_pred, Y_std = gp.predict(X_pred, return_std=True)

        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0], c='r', ls='-')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] + Y_std, c='r', ls='--')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] - Y_std, c='r', ls='--')
        axes[ix].plot(X_pred[:, 2], Y_pred[:, 0] + np.sqrt(40)*Y_std, c='g')
        fs = 8
        axes[ix].set_xlabel('q', fontsize=fs)
        axes[ix].set_ylabel('d', fontsize=fs)
        axes[ix].set_ylim(0, 0.1)
        x0, x1 = axes[ix].get_xlim()
        y0, y1 = axes[ix].get_ylim()
        axes[ix].set_aspect((x1 - x0) / (y1 - y0))
        axes[ix].set(adjustable='box')
        axes[ix].set_title('pitch=%.2f' % pitch, fontsize=fs)
    plt.show()
    #plt.savefig('gp_estimates_%d.png' % num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int)
    args = parser.parse_args()

    with open('prism_gp_1.pickle', 'rb') as handle:
        data = pickle.load(handle)

    ucb_interaction(data[0], max_iterations=1000)
    X, Y, X_pred = process_data(data, n_train=args.n_train)

    kernel = ConstantKernel(1.0) * RBF(length_scale=(1.0, 1.0, 1.0), length_scale_bounds=(1e-5, 1)) + WhiteKernel(noise_level=0.01)
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


