 # read in a learned model and predict for new busybox
import argparse
import operator
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import BusyBox, Slider, Door
from util.setup_pybullet import setup_env
from scipy.optimize import minimize
from learning.dataloaders import parse_pickle_file, PolicyDataset
import torch
from util import util, plot_results
import numpy as np
import sys
import matplotlib.pyplot as plt
import pybullet as p
from torch.utils.tensorboard import SummaryWriter
import os

def vis_test_error(fname):
    ntrain = 10000
    step = 1000
    bbs = util.read_from_file(fname)
    data_types = ['active', 'random']
    plt.ion()
    plot_obj = plot_results.TestMechPoliciesPitchOnly()
    ns = range(step, ntrain+1, step)

    bbs = bbs[:7]
    for train_data_type in data_types:
        for val_data_type in data_types:
            for n in ns:
                net_file = 'torch_models_2019-08-28_15-54-10/'+train_data_type+'/'+val_data_type+'/'+str(n)+'.pt'
                net = util.load_model(net_file, hdim=16)
                plot_obj._plot(None, model=net, bbps=bbs)
                dir = 'test_plots/'+train_data_type+'/'+val_data_type
                file = str(n)+'.png'
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path = dir + '/' + file
                plt.savefig(path, bbox_inches='tight')
                #plt.show()
                #input()
                #plt.close()

def calc_test_error(fname):
    ntrain = 10000
    step = 1000
    bbs = util.read_from_file(fname)
    data_types = ['active', 'random']
    writers = {}
    ns = range(step, ntrain+1, step)
    for train_data_type in data_types:
        for val_data_type in data_types:
            writer_key = train_data_type+','+val_data_type
            writers[writer_key] = SummaryWriter('./test_runs/'+writer_key)
            for n in ns:
                test_error = 0
                for bbp in bbs:
                    # make new bb object from pickle file
                    rand_num = np.random.uniform(0,1)
                    bb = BusyBox.get_busybox(bbp.width, bbp.height, bbp._mechanisms, urdf_tag=str(rand_num))
                    net_file = 'torch_models_2019-08-28_15-54-10/'+train_data_type+'/'+val_data_type+'/'+str(n)+'.pt'
                    net = util.load_model(net_file, hdim=16)
                    true_motion = bb._mechanisms[0].range/2
                    test_motion = test_env(net, bb=bb, debug=False, plot=False, viz=False)
                    test_error += np.linalg.norm([true_motion-test_motion])**2
                test_mse = test_error/len(bbs)
                writers[writer_key].add_scalar('test_error', test_mse, n)
                print(writer_key, test_mse, n)
    for writer in writers.values():
        writer.close()

def calc_true_error():
    ntrain = 10000
    step = 1000
    bbs = util.read_from_file('data/datasets/40_bb_sliders.pickle')
    data_types = ['active', 'random']
    writers = {}
    ns = range(step, ntrain+1, step)
    for train_data_type in data_types:
        for val_data_type in data_types:
            writer_key = train_data_type+','+val_data_type
            writers[writer_key] = SummaryWriter('./test_true_runs/'+writer_key)
            for n in ns:
                test_error = 0
                for bbp in bbs:
                    # make new bb object from pickle file
                    rand_num = np.random.uniform(0,1)
                    bb = BusyBox.get_busybox(bbp.width, bbp.height, bbp._mechanisms, urdf_tag=str(rand_num))
                    mech = bb._mechanisms[0]
                    image_data = setup_env(bb, False, False)
                    net_file = 'torch_models_2019-08-28_15-54-10/'+train_data_type+'/'+val_data_type+'/'+str(n)+'.pt'
                    net = util.load_model(net_file, hdim=16)
                    mech_tuple = mech.get_mechanism_tuple()
                    true_policy = policies.generate_policy(bb, mech, True, 0.0)
                    policy_type = true_policy.type
                    q = true_policy.generate_config(mech, 1.0)
                    policy_tuple = true_policy.get_policy_tuple()
                    results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                            None, None, q, image_data, None, 1.0)]
                    pred_motion = get_pred_motions(results, net, ret_dataset=False, use_cuda=False)[0]
                    true_motion = bb._mechanisms[0].range/2
                    #test_motion = test_env(net, bb=bb, debug=False, plot=False, viz=False)
                    #test_error += np.linalg.norm([true_motion-test_motion])**2
                    test_error += np.linalg.norm([true_motion-pred_motion])**2
                test_mse = test_error/len(bbs)
                writers[writer_key].add_scalar('test_error', test_mse, n)
                print(writer_key, test_mse, n)
    for writer in writers.values():
        writer.close()

def get_pred_motions(data, model, ret_dataset=False, use_cuda=False):
    data = parse_pickle_file(data=data)
    dataset = PolicyDataset(data)
    pred_motions = []
    for i in range(len(dataset.items)):
        policy_type = dataset.items[i]['type']
        policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
        policy_tensor = dataset.tensors[i].unsqueeze(0)
        config_tensor = dataset.configs[i].unsqueeze(0)
        image_tensor = dataset.images[i].unsqueeze(0)
        if use_cuda:
            policy_type_tensor = policy_type_tensor.cuda()
            policy_tensor = policy_tensor.cuda()
            config_tensor = config_tensor.cuda()
            image_tensor = image_tensor.cuda()
        pred_motion = model.forward(policy_type_tensor,
                                    policy_tensor,
                                    config_tensor,
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

def objective_func(x, policy_type, image_tensor, model, use_cuda):
    policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
    x = x.squeeze()
    policy_tensor = torch.tensor([[x[0], 0.0]]).float() # hard code yaw to be 0
    config_tensor = torch.tensor([[x[-1]]]).float()
    if use_cuda:
        policy_type_tensor = policy_type_tensor.cuda()
        policy_tensor = policy_tensor.cuda()
        config_tensor = config_tensor.cuda()
        image_tensor = image_tensor.cuda()
    val = -model.forward(policy_type_tensor, policy_tensor, config_tensor, image_tensor)
    if use_cuda:
        val = val.cpu()
    val = val.detach().numpy()
    val = val.squeeze()
    return val

def test_env(model, bb=None, plot=False, viz=False, debug=False, use_cuda=False):
    if bb is None:
        bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider])
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()

    n_samples = 500
    samples = []
    for _ in range(n_samples):
        random_policy = policies.generate_policy(bb, mech, True, 1.0)
        policy_type = random_policy.type
        q = random_policy.generate_config(mech, None)
        policy_tuple = random_policy.get_policy_tuple()
        results = [util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                None, None, q, image_data, None, 1.0)]
        sample_disps, dataset = get_pred_motions(results, model, ret_dataset=True, use_cuda=use_cuda)
        samples.append(((policy_type,
                        dataset.tensors[0].detach().numpy(),
                        dataset.configs[0].detach().numpy(),
                        policy_tuple.delta_values.delta_yaw,
                        policy_tuple.delta_values.delta_pitch),
                        sample_disps[0]))

    (policy_type_max, params_max, q_max, delta_yaw_max, delta_pitch_max), max_disp = max(samples, key=operator.itemgetter(1))
    pose_handle_base_world = mech.get_pose_handle_base_world()
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+list(params_max)
    # start optimization from here
    # assume you guessed the correct policy type, and optimize for params and configuration
    x0 = np.concatenate([[params_max[0]], q_max]) # only searching space of pitches!
    res = minimize(fun=objective_func, x0=x0, args=(policy_type_max, dataset.images[0].unsqueeze(0), model, use_cuda),
                method='BFGS', options={'eps': 10**-3}) # TODO: for some reason get pytorch error when change options
    x_final = res['x']
    policy_list = list(pose_handle_base_world.p)+list(pose_handle_base_world.q)+[x_final[0], 0.0] # hard code yaw value
    policy_final = policies.get_policy_from_params(policy_type_max, policy_list, mech)
    config_final = x_final[-1]

    if plot:
        plot_search(bb, samples, q_max, delta_yaw_max, delta_pitch_max, policy_final, config_final, debug)

    # test found policy on busybox
    setup_env(bb, viz, debug=debug)
    mech = bb._mechanisms[0]
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy_final.generate_trajectory(pose_handle_base_world, config_final, False)
    gripper = Gripper(bb.bb_id)
    policy_type = policy_final.type
    _, motion, _ = gripper.execute_trajectory(traj, mech, policy_type, debug)
    p.disconnect()

    if plot:
        plt.close()
    return motion

def plot_search(bb, samples, q_max, delta_yaw_max, delta_pitch_max, policy_final, config_final, debug):
    image_data = setup_env(bb, viz=False, debug=debug)
    mech = bb._mechanisms[0]
    mech_tuple = mech.get_mechanism_tuple()
    limit = mech_tuple.params.range/2

    qs = [s[0][2] for s in samples]
    delta_pitches = [s[0][4] for s in samples]
    disps = [s[1] for s in samples]
    mind = min(disps)
    maxd = max(disps)

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.scatter(qs, np.multiply(180/np.pi,delta_pitches), c=disps, vmin=mind, vmax=maxd, s=6)
    fig.colorbar(im)
    ax.set_title('Limit ='+str(mech.range/2))
    ax.set_xlabel('q')
    ax.set_ylabel('delta_pitch (deg)')
    ax.plot(q_max, delta_pitch_max, 'gx')
    ax.plot(config_final, policy_final.delta_pitch, 'r.')
    plt.show()
    input('press enter to close plot')
    #plt.close()

def viz_final(bb, policy_final, config_final, debug):
    # test found policy on busybox
    setup_env(bb, True, debug=debug)
    mech = bb._mechanisms[0]
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy_final.generate_trajectory(pose_handle_base_world, config_final, True)
    gripper = Gripper(bb.bb_id)
    policy_type = policy_final.type
    gripper.execute_trajectory(traj, mech, policy_type, debug)
    p.disconnect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    #parser.add_argument('--plot', action='store_true')
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument('--n-test', type=int, default=1) # how many mechanisms do you want to test
    parser.add_argument('--model', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--hdim', type=int, default=16)
    #parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--mode', choices=['single', 'true', 'test', 'plots'], required=True)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.mode == 'single':
        model = util.load_model(args.model, args.hdim)
        np.random.seed(11)
        test_env(model, plot=args.viz)
    if args.mode == 'true':
        calc_true_error(args.fname)
    if args.mode == 'test':
        calc_test_error(args.fname)
    if args.mode == 'plots':
        vis_test_error(args.fname)
