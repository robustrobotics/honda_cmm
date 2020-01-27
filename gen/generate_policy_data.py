import sys
import argparse
from utils import util
import numpy as np
import argparse
import pybullet as p
from utils.setup_pybullet import setup_env, custom_bb_door, custom_bb_slider
from utils.util import read_from_file
from actions import policies
from gen.generator_busybox import Slider, Door, BusyBox

def generate_dataset(args, git_hash):
    bb_dataset = get_bb_dataset(args.bb_fname, args.n_bbs, args.mech_types, args.max_mech, args.urdf_num)
    if args.n_samples == 0:
        return bb_dataset

    results = []
    for (i, bb_results) in enumerate(bb_dataset):
        bb = BusyBox.bb_from_result(bb_results[0])
        image_data, gripper = setup_env(bb, args.viz, args.debug, args.no_gripper)
        bb_results = []
        for j in range(args.n_samples):
            sys.stdout.write("\rProcessing sample %i/%i for busybox %i/%i" % (j+1, args.n_samples, i+1, args.n_bbs))
            for mech in bb._mechanisms:
                # generate either a random or model-based policy and goal configuration
                pose_handle_base_world = mech.get_pose_handle_base_world()
                policy = policies.generate_policy(bb, mech, args.random_policies, init_pose=pose_handle_base_world)
                pose_handle_world_init = mech.get_handle_pose()

                # calculate trajectory
                traj = policy.generate_trajectory(pose_handle_base_world, args.debug, color=[0, 0, 1])

                # execute trajectory
                cumu_motion, net_motion, pose_handle_world_final = \
                        gripper.execute_trajectory(traj, mech, policy.type, args.debug)
                # save result data
                policy_params = policy.get_policy_tuple()
                mechanism_params = mech.get_mechanism_tuple()
                bb_results.append(util.Result(policy_params, mechanism_params, net_motion, \
                            cumu_motion, pose_handle_world_init, pose_handle_world_final, \
                            image_data, git_hash, args.no_gripper))

                gripper.reset(mech)
        results.append(bb_results)
        p.disconnect()
    print()
    return results

def get_bb_dataset(bb_fname, n_bbs, mech_types, max_mech, urdf_num):
    # Create a dataset of busyboxes.
    if bb_fname == '' or bb_fname is None:
        print('Creating Busyboxes.')
        mech_classes = []
        for mech_type in mech_types:
            if mech_type == 'slider': mech_classes.append(Slider)
            if mech_type == 'door': mech_classes.append(Door)

        bb_dataset = []
        for _ in range(n_bbs):
            # TODO: i think there is a bug here...
            bb = BusyBox.generate_random_busybox(max_mech=max_mech,
                                                    mech_types=mech_classes,
                                                    urdf_tag=urdf_num)
            mechanism_params = bb._mechanisms[0].get_mechanism_tuple()
            image_data, gripper = setup_env(bb, False, False, True)
            bb_dataset.append([util.Result(None, mechanism_params, None, None, None,
                                None, image_data, None, None)])
        print('BusyBoxes created.')
    else:
        # Load in a file with predetermined BusyBoxes.
        bb_dataset = read_from_file(bb_fname)
    return bb_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=1) # number samples per bb
    parser.add_argument('--n-bbs', type=int, default=5) # number bbs to generate
    parser.add_argument('--max-mech', type=int, default=1) # mechanisms per bb
    parser.add_argument('--mech-types', nargs='+', default=['slider'], type=str)
    parser.add_argument('--fname', type=str) # give filename if want to save to file
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--random-policies', action='store_true') # if want to only use random policy class on mechanisms
    # desired goal config represented as a percentage of the max config, if unused then random config is generated
    parser.add_argument('--bb-fname', type=str)
    parser.add_argument('--no-gripper', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        # write git has to results if have package
        import git
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
    except:
        print('install gitpython to save git hash to results')
        git_hash = None
    results = generate_dataset(args, git_hash)
    if args.fname:
        util.write_to_file(args.fname, results)
