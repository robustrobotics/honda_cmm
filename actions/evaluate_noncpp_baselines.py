import argparse
import pickle

import pybullet as p

from learning.gp.evaluate_models import SUCCESS_REGRET
from learning.gp.explore_single_bb import create_single_bb_gpucb_dataset
from gen.generate_policy_data import get_bb_dataset
from gen.generator_busybox import BusyBox
from actions import policies
from utils.setup_pybullet import setup_env
from utils import util

def main(args):
    busybox_data = get_bb_dataset(args.bb_fname, args.N, args.mech_types, 1, args.urdf_num)
    all_steps = []
    for ix, bb_results in enumerate(busybox_data):
        if args.type == 'gpucb':
            single_dataset, _, steps = create_single_bb_gpucb_dataset(bb_results[0],
                                            '',
                                            args.plot,
                                            args,
                                            ix,
                                            success_regret=SUCCESS_REGRET)
            all_steps.append(steps)
            print('steps', steps)
        elif args.type == 'random':
            bb = BusyBox.bb_from_result(bb_results[0])
            image_data, gripper = setup_env(bb, args.viz, args.debug, not args.use_gripper)
            regret = float("inf")
            steps = 0
            while regret > SUCCESS_REGRET:
                mech = bb._mechanisms[0]
                # generate either a random or model-based policy and goal configuration
                policy = policies.generate_policy(mech, args.random_policies)
                pose_handle_world_init = mech.get_handle_pose()

                # calculate trajectory
                pose_handle_base_world = mech.get_pose_handle_base_world()
                traj = policy.generate_trajectory(pose_handle_base_world, args.debug, color=[0, 0, 1])

                # execute trajectory
                cumu_motion, net_motion, pose_handle_world_final = \
                        gripper.execute_trajectory(traj, mech, policy.type, args.debug)
                
                # calc regret
                max_dist = mech.get_max_net_motion()
                regret = (max_dist - net_motion)/max_dist
                steps += 1
                
                # reset
                gripper.reset(mech)
            p.disconnect()
            all_steps.append(steps)
            print('steps', steps)

    # Save the dataset.
    util.write_to_file(args.fname, all_steps)
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--use-gripper', action='store_true')
    parser.add_argument('--fname', type=str)
    parser.add_argument(
        '--n-gp-samples',
        type=int,
        default=1000,
        help='number of samples to use when fitting a GP to data')
    parser.add_argument(
        '--N',
        type=int,
        help='number of BusyBoxes to interact with during evaluation time')
    parser.add_argument(
        '--type',
        type=str,
        required=True,
        help='evaluation type in [random, gpucb]')
    parser.add_argument(
        '--mech-types',
        nargs='+',
        default=['slider'],
        type=str,
        help='if no bb-fname is specified, list the mech types desired')
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
        '--random-policies',
        action='store_true',
        help='use to try random policy classes on random mechanisms')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='sample from acquistion function')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()
        
    main(args)
