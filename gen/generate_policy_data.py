import sys
import argparse
from util import util
import numpy as np
import argparse
import pybullet as p
from util.setup_pybullet import setup_env, custom_bb_door, custom_bb_slider
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import Slider, Door, BusyBox

results = []

def reset(bb, viz, debug):
    setup_env(bb, viz, debug)
    return Gripper(bb.bb_id)

def generate_dataset(args, git_hash, bb=None):
    for i in range(args.n_bbs):
        if bb is None:
            # TODO: add back doors when working
            bb = BusyBox.generate_random_busybox(max_mech=args.max_mech, mech_types=[Slider], urdf_tag=str(args.urdf_num), debug=args.debug)

        image_data = setup_env(bb, args.viz, args.debug)
        for j in range(args.n_samples):
            sys.stdout.write("\rProcessing sample %i/%i for busybox %i/%i" % (j+1, args.n_samples, i+1, args.n_bbs))
            # setup env and get image before load gripper
            gripper = reset(bb, args.viz, args.debug)
            for mech in bb._mechanisms:
                # generate either a random or model-based policy and goal configuration
                policy = policies.generate_policy(bb, mech, args.match_policies, args.randomness)
                config_goal = policy.generate_config(mech, args.goal_config)
                pose_handle_world_init = util.Pose(*p.getLinkState(bb.bb_id, mech.handle_id)[:2])

                # calculate trajectory
                pose_handle_base_world = mech.get_pose_handle_base_world()
                traj = policy.generate_trajectory(pose_handle_base_world, config_goal, args.debug)

                # execute trajectory
                cumu_motion, net_motion, pose_handle_world_final = \
                        gripper.execute_trajectory(traj, mech, policy.type, args.debug)

                # save result data
                policy_params = policy.get_policy_tuple()
                mechanism_params = mech.get_mechanism_tuple()
                results.append(util.Result(policy_params, mechanism_params, net_motion, \
                            cumu_motion, pose_handle_world_init, pose_handle_world_final, \
                            config_goal, image_data, git_hash, args.randomness))

    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=1) # number samples per bb
    parser.add_argument('--n-bbs', type=int, default=5) # number bbs to generate
    parser.add_argument('--max-mech', type=int, default=1) # mechanisms per bb
    parser.add_argument('--fname', type=str) # give filename if want to save to file
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--match-policies', action='store_true') # if want to only use correct policy class on mechanisms
    parser.add_argument('--randomness', type=float, default=1.0) # how far from true policy parameters to sample (as a fraction)
    # desired goal config represented as a percentage of the max config, if unused then random config is generated
    parser.add_argument('--goal-config', type=float)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        try:
            # write git has to results if have package
            import git
            repo = git.Repo(search_parent_directories=True)
            git_hash = repo.head.object.hexsha
        except:
            print('install gitpython to save git hash to results')
            git_hash = None
        generate_dataset(args, git_hash)
        if args.fname:
            util.write_to_file(args.fname, results)
    except KeyboardInterrupt:
        # if Ctrl+C write to pickle
        if args.fname:
            util.write_to_file(args.fname, results)
        print('Exiting...')
    except:
        # if crashes write to pickle
        if args.fname:
            util.write_to_file(args.fname, results)

        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
