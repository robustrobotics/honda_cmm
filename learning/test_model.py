# read in a learned model and predict for new busybox
import argparse
import operator
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import BusyBox
from setup_pybullet import setup_env
from scipy.optimize import minimize
from learning.nn_disp_pol import DistanceRegressor as NNPol
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis
from learning.dataloaders import parse_pickle_file, PolicyDataset
import torch
import torchvision.transforms as transforms
from util import util
import numpy as np
import pybullet as p
import pybullet_data

def objective_func(x, policy_type, image_tensor, model):
    return -model.forward(policy_type, torch.tensor([x[:-1]]).float(), torch.tensor([[x[-1]]]).float(), image_tensor)

def test_random_env(model, viz, debug):
    if args.model == 'pol':
        model = NNPol(policy_names=['Prismatic', 'Revolute'],
                    policy_dims=[10, 14],
                    hdim=16)
        model.load_state_dict(torch.load('data/pol_model.pt'))
    else:
        model = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                       policy_dims=[10, 14],
                       hdim=16,
                       im_h=154,
                       im_w=205,
                       kernel_size=5)
        model.load_state_dict(torch.load('data/polvis_model.pt'))
    model.eval()

    # only testing with 1 Mechanism per BusyBox right now
    bb = BusyBox.generate_random_busybox(max_mech=1)
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    n_samples = 100
    samples = []

    for _ in range(n_samples):
        random_policy = policies.generate_random_policy(bb, mech)
        policy_type = random_policy.type
        q = random_policy.generate_random_config()
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(None, policy_tuple, None, None, 0., None, None, q, image_data, None)]
        data = parse_pickle_file(data=results)
        dataset = PolicyDataset(data)
        sample_disp = model.forward(policy_type,
                                        dataset.tensors[0].unsqueeze(0),
                                        dataset.configs[0].unsqueeze(0),
                                        dataset.images[0])
        samples.append(((policy_type, data[0]['params'], data[0]['config']), sample_disp))

    (policy_type, params_max, q_max), max_disp = max(samples, key=operator.itemgetter(1))

    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.array(params_max + [q_max])
    res = minimize(fun=objective_func, x0=x0, args=(policy_type, dataset.images[0], model),
                method='L-BFGS-B') #options={'eps': 10**-3})
    x_final = res['x']

    # test on busybox
    policy_final = policies.get_policy(policy_type, x_final[:-1])
    config_final = x_final[-1]

    # calculate trajectory
    p_handle_base_world = mech.get_pose_handle_base_world().p
    traj = policy_final.generate_trajectory(p_handle_base_world, config_final, debug)

    # execute trajectory
    setup_env(bb, viz=viz, debug=debug)
    gripper = Gripper(bb.bb_id)
    gripper.set_control_params(policy_final)
    gripper.execute_trajectory(traj, mech, debug=debug)
    p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', choices=['pol', 'polvis'], default='pol')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    test_random_env(args.model, args.viz, args.debug)
