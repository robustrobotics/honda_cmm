# read in a learned model and predict for new busybox
import argparse
import operator
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import BusyBox
from util.setup_pybullet import setup_env
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
from collections import namedtuple
import sys

SearchResult = namedtuple('SearchResult', 'mechanism image_data samples start_sample end_sample')
SampleResult = namedtuple('SampleResult', 'policy config_goal pred_motion')

def objective_func(x, policy_type, image_tensor, model):
    return -model.forward(policy_type, torch.tensor([x[:-1]]).float(), torch.tensor([[x[-1]]]).float(), image_tensor)

def test_random_env(model, viz, debug):
    # only testing with 1 Mechanism per BusyBox right now
    bb = BusyBox.generate_random_busybox(max_mech=1)
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    n_samples = 500
    samples = []
    sample_results = []

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
                                        dataset.images[0].unsqueeze(0))
        samples.append(((policy_type, data[0]['params'], data[0]['config']), sample_disp))
        sample_results.append(SampleResult(policy_tuple, q, sample_disp.detach().numpy()))

    (policy_type_max, params_max, q_max), max_disp = max(samples, key=operator.itemgetter(1))
    start_sample = SampleResult(policies.get_policy(policy_type_max, params_max).get_policy_tuple(), q_max, None)
    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.array(params_max + [q_max])
    res = minimize(fun=objective_func, x0=x0, args=(policy_type_max, dataset.images[0].unsqueeze(0), model),
                method='L-BFGS-B') #options={'eps': 10**-3})
    x_final = res['x']

    # test found policy on busybox
    setup_env(bb, viz=viz, debug=debug)
    policy_final = policies.get_policy(policy_type_max, x_final[:-1])
    end_sample = SampleResult(policy_final.get_policy_tuple(), x_final[-1], None)
    config_final = x_final[-1]
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy_final.generate_trajectory(pose_handle_base_world, config_final, True)
    gripper = Gripper(bb.bb_id)
    result = gripper.execute_trajectory(traj, mech, policy_type_max, False, debug=debug)

    # get what actual max disp is
    setup_env(bb, viz=viz, debug=debug)
    policy_truth = policies.generate_model_based_policy(bb, mech)
    config_truth = policy_truth.generate_model_based_config(mech)
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy_truth.generate_trajectory(pose_handle_base_world, config_truth, True)
    gripper = Gripper(bb.bb_id)
    result_truth = gripper.execute_trajectory(traj, mech, policy_truth.type, False, debug=debug)
    p.disconnect()
    return SearchResult(mech.get_mechanism_tuple(), image_data, sample_results, start_sample, end_sample)

def test_random_envs(n_test, model_type, file_name, viz, debug, use_cuda):
    if model_type == 'pol':
        model = NNPol(policy_names=['Prismatic', 'Revolute'],
                    policy_dims=[10, 14],
                    hdim=16)
    else:
        model = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                       policy_dims=[10, 14],
                       hdim=16,
                       im_h=154,
                       im_w=205,
                       kernel_size=5)
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(file_name, map_location=device))
    model.eval()

    search_results = []
    for i in range(n_test):
        sys.stdout.write("\rProcessing mechanism %i/%i" % (i+1, n_test))
        search_results.append(test_random_env(model, viz, debug))
    print()
    return search_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model-type', choices=['pol', 'polvis'], default='polvis')
    parser.add_argument('--n-test', type=int, default=10) # how many mechanisms do you want to test
    parser.add_argument('--model-fname', type=str)
    parser.add_argument('--results-fname', type=str)
    parser.add_argument('--use-cuda', default=False)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    file_name = args.model_fname + '.pt'
    search_results = test_random_envs(args.n_test, args.model_type, file_name, args.viz, args.debug, args.use_cuda)

    if args.results_fname:
        util.write_to_file(args.results_fname, search_results)
