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
from learning.train import name_lookup
import matplotlib.pyplot as plt
import pybullet as p

def get_pred_motions(data, model, ret_dataset=False):
    data = parse_pickle_file(data=data)
    dataset = PolicyDataset(data)
    pred_motions = []
    for i in range(len(dataset.items)):
        policy_type = dataset.items[i]['type']
        pred_motion = model.forward(torch.Tensor([name_lookup[policy_type]]),
                                    dataset.tensors[i].unsqueeze(0),
                                    dataset.configs[i].unsqueeze(0),
                                    dataset.images[i].unsqueeze(0))
        pred_motion_float = pred_motion.detach().numpy()[0][0]
        pred_motions += [pred_motion_float]
    if ret_dataset:
        return pred_motions, dataset
    else:
        return pred_motions

def objective_func(x, policy_type, image_tensor, model):
    policy_type_tensor = torch.Tensor([name_lookup[policy_type]])
    return -model.forward(policy_type_tensor, torch.tensor([x[:-1]]).float(), torch.tensor([[x[-1]]]).float(), image_tensor)

def test_random_env(model, viz, debug):
    bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider])
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()
    limit = mech_tuple.params.range/2
    n_samples = 500
    samples = []

    n_q_bins = 6
    qs = np.linspace(0.0, limit*1.2, n_q_bins+1)
    plot_data = {}
    for _ in range(n_samples):
        random_policy = policies.generate_policy(bb, mech, True, 1.0)
        policy_type = random_policy.type
        q = random_policy.generate_config(mech, None)
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                None, None, q, image_data, None, 1.0)]
        sample_disps, dataset = get_pred_motions(results, model, ret_dataset=True)
        closest_q = min(qs, key=lambda x:abs(x-abs(q)))
        closest_q_i = list(qs).index(closest_q)
        if closest_q > abs(q) or closest_q_i == n_q_bins:
            closest_q_i -= 1
        if closest_q_i not in plot_data:
            plot_data[closest_q_i] = [[policy_tuple.delta_values.delta_yaw],
                                        [policy_tuple.delta_values.delta_pitch],
                                        [sample_disps[0]]]
        else:
            plot_data[closest_q_i][0] += [policy_tuple.delta_values.delta_yaw]
            plot_data[closest_q_i][1] += [policy_tuple.delta_values.delta_pitch]
            plot_data[closest_q_i][2] += [sample_disps[0]]
        samples.append(((policy_type,
                        dataset.tensors[0].detach().numpy(),
                        dataset.configs[0].detach().numpy(),
                        policy_tuple.delta_values.delta_yaw,
                        policy_tuple.delta_values.delta_pitch),
                        sample_disps[0]))
    #print(samples)
    (policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp = max(samples, key=operator.itemgetter(1))
    #print((policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp)
    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+list(params_max)
    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.concatenate([params_max, q_max])
    res = minimize(fun=objective_func, x0=x0, args=(policy_type_max, dataset.images[0].unsqueeze(0), model),
                method='BFGS')#, options={'eps': 10**-3}) # TODO: for some reason get pytorch error when change options
    x_final = res['x']

    # visualize samples and minimization in parameter space
    min_motion = float('inf')
    max_motion = 0.0
    for i in range(n_q_bins):
        if i in plot_data:
            if min(plot_data[i][2])<min_motion:
                min_motion = min(plot_data[i][2])
            if max(plot_data[i][2])>max_motion:
                max_motion = max(plot_data[i][2])
    plt.ion()
    fig = plt.figure()
    axes = fig.subplots(n_q_bins, 1)
    plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
    for j in range(n_q_bins):
        if j == n_q_bins-1:
            axes[j].set_title('q>['+str(round(qs[j],2)))
        else:
            axes[j].set_title('q=['+str(round(qs[j],2))+','+str(round(qs[j+1],2))+']')
        if j in plot_data:
            im = axes[j].scatter(plot_data[j][0], plot_data[j][1], c=plot_data[j][2], vmin=min_motion, vmax=max_motion)
            axes[j].tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle('Limit='+str(round(limit, 2)), fontsize=16)

    # add optimization start and final sample to plot
    closest_q = min(qs, key=lambda x:abs(x-abs(q_max)))
    closest_q_i = list(qs).index(closest_q)
    if closest_q > abs(q_max) or closest_q_i == n_q_bins:
        closest_q_i -= 1
    axes[closest_q_i].plot(delta_yaw_max, delta_pitch_max, 'gx')

    # test found policy on busybox
    setup_env(bb, viz=viz, debug=debug)
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+list(x_final[:-1])
    policy_final = policies.get_policy_from_params(policy_type_max, policy_list, mech)
    config_final = x_final[-1]

    closest_q = min(qs, key=lambda x:abs(x-abs(config_final)))
    closest_q_i = list(qs).index(closest_q)
    if closest_q > abs(config_final) or closest_q_i == n_q_bins:
        closest_q_i -= 1
    axes[closest_q_i].plot(policy_final.delta_yaw, policy_final.delta_pitch, 'r.')
    plt.show()
    if not viz:
        input('press enter to close plot')

    traj = policy_final.generate_trajectory(pose_handle_base_world, config_final, True)
    gripper = Gripper(bb.bb_id)
    gripper.execute_trajectory(traj, mech, policy_type_max, debug)
    if viz:
        try:
            while True:
                # so can look at trajectory and interact with sim
                p.stepSimulation()
        except KeyboardInterrupt:
            p.disconnect()

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
