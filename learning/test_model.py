# read in a learned model and predict for new busybox
import argparse
import operator
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import BusyBox, Slider, Door
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


def get_pred_motions(data, model, ret_dataset=False):
    data = parse_pickle_file(data=data)
    dataset = PolicyDataset(data)
    pred_motions = []
    for i in range(len(dataset.items)):
        policy_type = dataset.items[i]['type']
        policy_params = dataset.tensors[i].unsqueeze(0)
        pred_motions += [model.forward(policy_type,
                                    policy_params,
                                    dataset.configs[i].unsqueeze(0),
                                    dataset.images[i].unsqueeze(0))]
    if ret_dataset:
        return pred_motions, dataset
    else:
        return pred_motions

def objective_func(x, policy_type, image_tensor, model):
    return -model.forward(policy_type, torch.tensor([x[:-1]]).float(), torch.tensor([[x[-1]]]).float(), image_tensor)

def test_random_env(model, viz, debug):
    bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider])
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()
    n_samples = 500
    samples = []
    sample_results = []

    for _ in range(n_samples):
        random_policy = policies.generate_policy(bb, mech, True, 1.0)
        policy_type = random_policy.type
        q = random_policy.generate_config(mech, None)
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                None, None, q, image_data, None, 1.0)]
        sample_disps, dataset = get_pred_motions(results, model, ret_dataset=True)
        samples.append(((policy_type,
                            dataset.tensors[0].detach().numpy(),
                            dataset.configs[0].detach().numpy()),
                            sample_disps[0].detach().numpy()))
        sample_results.append(SampleResult(policy_tuple, q, sample_disps[0].detach().numpy()))

    (policy_type_max, params_max, q_max), max_disp = max(samples, key=operator.itemgetter(1))
    start_sample = SampleResult(policies.get_policy_from_params(policy_type_max, params_max).get_policy_tuple(), q_max, None)
    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.concatenate([params_max, q_max])
    res = minimize(fun=objective_func, x0=x0, args=(policy_type_max, dataset.images[0].unsqueeze(0), model),
                method='L-BFGS-B') #options={'eps': 10**-3})
    x_final = res['x']

    # test found policy on busybox
    setup_env(bb, viz=viz, debug=debug)
    policy_final = policies.get_policy_from_params(policy_type_max, x_final[:-1])
    end_sample = SampleResult(policy_final.get_policy_tuple(), x_final[-1], None)
    config_final = x_final[-1]
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy_final.generate_trajectory(pose_handle_base_world, config_final, True)
    gripper = Gripper(bb.bb_id)
    gripper.execute_trajectory(traj, mech, policy_type_max, debug)

def test_random_envs(n_test, file_name, hdim, viz, debug):
    model = util.load_model(file_name, hdim)

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
    parser.add_argument('--n-test', type=int, default=1) # how many mechanisms do you want to test
    parser.add_argument('--model-fname', type=str)
    parser.add_argument('--search-fname', type=str)
    parser.add_argument('--hdim', type=int, default=16)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    search_results = test_random_envs(args.n_test, args.model_fname, args.hdim, args.viz, args.debug)

    if args.search_fname:
        util.write_to_file(args.search_fname, search_results)
