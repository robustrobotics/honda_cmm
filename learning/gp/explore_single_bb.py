from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Sum
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


def objective_func(x, model):
    x = x.squeeze()
    print(x)

    X = np.array([[x[0], 0.0, x[-1]]])
    Y = -model.predict(X)

    return Y[0]

def get_pred_motions(data, model):
    X, Y, _ = process_data(data, len(data))
    return model.predict(X)


def optimize_gp(gp, result):
    n_samples = 500
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
        sample_disps = get_pred_motions(results, gp)


        samples.append(((policy_type,
                        policy_tuple,
                        q,
                        policy_tuple.delta_values.delta_yaw,
                        policy_tuple.delta_values.delta_pitch),
                        sample_disps[0]))

    # Find the sample that maximizes the distance.
    (policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp = max(samples, key=operator.itemgetter(1))

    # Start optimization from here.
    print(params_max)
    x0 = np.concatenate([[params_max.params.pitch], [q_max]]) # only searching space of pitches!
    print(x0)
    res = minimize(fun=objective_func, x0=x0, args=(gp,),
                   method='L-BFGS-B', options={'eps': 10**-3}, bounds=[(-np.pi, 0),(-0.5,0.5)]) # TODO: for some reason get pytorch error when change options
    x_final = res['x']

    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+[x_final[0], 0.0] # hard code yaw value

    return x_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-train', type=int)
    args = parser.parse_args()

    with open('prism_gp_4.pickle', 'rb') as handle:
        data = pickle.load(handle)

    X, Y, X_pred = process_data(data, n_train=args.n_train)

    kernel = Sum(RBF(length_scale=10.0, length_scale_bounds=(0, 10)),
                 WhiteKernel(noise_level=0.01))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(X, Y)

    max_x = optimize_gp(gp, data[0])

    Y_pred, Y_std = gp.predict(X_pred, return_std=True)

    plt.plot(X_pred[:, 0], Y_pred, c='r', ls='-')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] + Y_std, c='r', ls='--')
    plt.plot(X_pred[:, 0], Y_pred[:, 0] - Y_std, c='r', ls='--')
    plt.scatter(X[:, 0], Y, c='b')
    max_y = gp.predict(np.array([[max_x[0], 0, max_x[1]]]))
    plt.scatter(max_x[0], max_y[0], c='g')
    plt.show()


