import sys
import argparse
from util import util
import numpy as np
import argparse
import pybullet as p
from util.setup_pybullet import setup_env
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import Slider, Door, BusyBox

def generate_samples(viz, debug, max_mech, match_policies, randomness,
                            go_to_limit, git_hash=None, tag=''):
    bb = BusyBox.generate_random_busybox(max_mech=max_mech, mech_types=[Door, Slider], urdf_tag=tag, debug=debug)

    # setup env and get image before load gripper
    image_data = setup_env(bb, viz, debug)
    gripper = Gripper(bb.bb_id)

    results = []
    for mech in bb._mechanisms:
        # generate either a random or model-based policy and goal configuration
        policy = policies.generate_policy(bb, mech, match_policies, randomness)
        config_goal = policy.generate_config(mech, go_to_limit)
        pose_handle_world_init = p.getLinkState(bb.bb_id, mech.handle_id)[:2]

        # calculate trajectory
        pose_handle_base_world = mech.get_pose_handle_base_world()
        traj = policy.generate_trajectory(pose_handle_base_world, config_goal, debug)

        # execute trajectory
        joint_motion, pose_handle_world_final = \
                gripper.execute_trajectory(traj, mech, policy.type, debug)

        # save result data
        policy_params = policy.get_policy_tuple()
        mechanism_params = mech.get_mechanism_tuple()
        results += [util.Result(policy_params, mechanism_params, joint_motion, \
                    pose_handle_world_init, pose_handle_world_final, \
                    config_goal, image_data, git_hash, randomness)]

    p.disconnect()
    return results

results = []
def generate_dataset(n_samples, viz, debug, git_hash, urdf_num, match_policies, \
                        randomness, go_to_limit, max_mech):
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        results.extend(generate_samples(viz, debug, max_mech, match_policies, randomness,
                                    go_to_limit, git_hash, urdf_num))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5) # number bbs to generate
    parser.add_argument('--max-mech', type=int, default=1) # mechanisms per bb
    parser.add_argument('--fname', type=str, required=True) # give filename
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--match-policies', action='store_true')
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--go-to-limit', action='store_true')
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
        generate_dataset(args.n_samples, args.viz, args.debug, git_hash, args.urdf_num, \
                        args.match_policies, args.randomness, args.go_to_limit, \
                        args.max_mech)
        util.write_to_file(args.fname, results)
    except KeyboardInterrupt:
        # if Ctrl+C write to pickle
        util.write_to_file(args.fname, results)
        print('Exiting...')
    except:
        # if crashes write to pickle
        util.write_to_file(args.fname, results)

        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
