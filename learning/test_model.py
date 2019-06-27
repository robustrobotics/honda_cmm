# read in a learned model and predict for new busybox
import argparse
import operator
from actions import policies
from setup_pybullet import random_env
from scipy.optimize import minimize
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
import torch

def objective_func(x, k, image, model):
    return -model.forward(k, x[0], x[1], image)

def test_random_env(model_path, viz, max_mech, debug):
    # TODO: ability to later visualize when testing max policy, but not during sampling phase
    bb, gripper, image_data = random_env(viz=False, max_mech=1, debug=debug)
    model = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[10, 14],
                   hdim=args.hdim,
                   im_h=154,
                   im_w=205,
                   kernel_size=5).cuda()
    model.load_state_dict(torch.load('data/model.pt'))
    model.eval()

    # only testing with 1 Mechanism per BusyBox right now
    mech = bb._mechanisms[0]
    n_samples = 100
    samples = []

    for _ in range(n_samples):
        random_policy = policies.generate_random_policy(bb, mech)
        k = random_policy.type
        theta = random_policy.get_params_tuple()
        q = random_policy.generate_random_config()
        image= image_data.rgbPixels
        samples += [[(k, theta, q, im), model.forward(k, theta, q, im)]]

    (k_max, theta_max, q_max, im), max_output = max(samples, key=operator.itemgetter(1))

    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.array(theta_max, q_max)
    res = minimize(fun=objective_func, x0=x0, args=(k_max, image, model),
                method='L-BFGS-B') #options={'eps': 10**-3})
    x_final = res['x']

    # test on busybox
    policy_final = generate_policy(k_max, x_final[0])
    config_final = x_final[1]
    gripper.set_control_params(policy_final)

    # calculate trajectory
    p_handle_base_world = mech.get_pose_handle_base_world().p
    traj = max_policy.generate_trajectory(p_handle_base_world, config_final, debug)

    # execute trajectory
    gripper.execute_trajectory(traj, mech, debug=debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-mech', type=int, default=1)
    parser.add_argument('--model-path', type=str)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    test_random_env(args.model_path, args.viz, args.max_mech, args.debug)
