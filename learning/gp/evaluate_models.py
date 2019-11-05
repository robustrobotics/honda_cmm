import pickle
import argparse
import os
import numpy as np
from learning.gp.explore_single_bb import create_single_bb_gpucb_dataset, GPOptimizer
from utils import util, setup_pybullet
from gen.generator_busybox import BusyBox
from actions.gripper import Gripper

def get_models(L, models_path):
    all_files = os.walk(models_path)
    models = []
    for root, subdir, files in all_files:
        for file in files:
            if file[-3:] == '.pt' and str(L)+'.pt' in file:
                full_path = root+'/'+file
                models.append(full_path)
    return models

def test_model(gp, bb_result, args, nn=None, use_cuda=False, urdf_num=0):
    """
    Maximize the GP mean function to get the best policy.
    :param gp: A GP fit to the current BusyBox.
    :param result: Result representing the current BusyBox.
    :return: Regret.
    """
    # Optimize the GP to get the best result.
    bb = BusyBox.bb_from_result(bb_result, urdf_num=urdf_num)
    image_data = setup_pybullet.setup_env(bb, viz=False, debug=False)
    optim_gp = GPOptimizer(urdf_num, bb, image_data, args.n_gp_samples, nn=nn)
    policy, q = optim_gp.optimize_gp(gp, ucb=False)

    # Execute the policy and observe the true motion.
    mech = bb._mechanisms[0]
    gripper = Gripper()
    pose_handle_base_world = mech.get_pose_handle_base_world()

    traj = policy.generate_trajectory(pose_handle_base_world, q, True)
    _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, False)
    # import time
    # time.sleep(1)
    #p.disconnect()

    # Calculate the regret.
    max_d = bb._mechanisms[0].get_max_dist()
    regret = (max_d - motion)/max_d

    return regret

def evaluate_models(n_interactions, n_bbs, args, use_cuda=False):
    with open(args.bb_fname, 'rb') as handle:
        bb_data = pickle.load(handle)

    all_results = {}
    for L in range(1000, 10001, 1000):
        models = get_models(L, args.models_path)
        all_L_results = {}
        for model in models:
            all_model_test_regrets = []
            for ix, bb_result in enumerate(bb_data[:n_bbs]):
                if args.debug:
                    print('BusyBox', ix)
                dataset, avg_regret, gp = create_single_bb_gpucb_dataset(bb_result, n_interactions, model, args.plot, args)
                nn = util.load_model(model, args.hdim, use_cuda=False)
                regret = test_model(gp, bb_result, args, nn, use_cuda=use_cuda, urdf_num=args.urdf_num)
                all_model_test_regrets.append(regret)
                if args.debug:
                    print('Test Regret   :', regret)
            if args.debug:
                print('Results')
                #print('Average Regret:', np.mean(avg_regrets))
                print('Final Regret  :', np.mean(all_model_test_regrets))
            all_L_results[model] = all_model_test_regrets
        if len(models) > 0:
            # TODO: this shouldn't be hard coded to 100 interactions per BB
            all_results[L/100] =  all_L_results
    util.write_to_file('regret_results_%s_%dT_%dN.pickle' % (args.type, n_interactions, n_bbs), all_results, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-gp-samples',
        type=int,
        default=500,
        help='number of samples to use when fitting a GP to data')
    parser.add_argument(
        '--T',
        type=int,
        help='number of interactions within a single BusyBox during evaluation time')
    parser.add_argument(
        '--N',
        type=int,
        help='number of BusyBoxes to interact with during evaluation time')
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        help='evaluation type in [random, gpucb, active, systematic]')
    parser.add_argument(
        '--urdf-num',
        default=0,
        help='number to append to generated urdf files. Use if generating multiple datasets simultaneously.')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='use to generate polar plots durin GP-UCB interactions')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--hdim',
        type=int,
        default=16,
        help='hdim of supplied model(s), used to load model file'
    )
    parser.add_argument(
        '--models-path',
        help='path to model files')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    evaluate_models(args.T, args.N, args, use_cuda=False)
