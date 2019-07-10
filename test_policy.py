from util import util
import numpy as np
import argparse
import pybullet as p
from util.setup_pybullet import setup_env
from actions import policies
from actions.gripper import Gripper
from gen.generator_busybox import Slider, Door, BusyBox

def test_policy(viz=False, debug=False, max_mech=6, random_policy=False, k=None, d=None,\
                    add_dist=None, p_err_thresh=None, p_delta=None, git_hash=None,\
                    tag=''):

    bb = BusyBox.generate_random_busybox(max_mech=max_mech, mech_types=[Slider], urdf_tag=tag, debug=debug)
    # setup env and get image before load gripper
    image_data = setup_env(bb, viz=viz, debug=debug)
    gripper = Gripper(bb.bb_id, k, d, add_dist, p_err_thresh)

    results = []
    for mech in bb._mechanisms:
        # generate either a random or model-based policy and goal configuration
        if random_policy:
            policy = policies.generate_random_policy(bb, mech, p_delta)
            config_goal = policy.generate_random_config()
        else:
            policy = policies.generate_model_based_policy(bb, mech, p_delta)
            config_goal = policy.generate_model_based_config(mech, go_to_limit=False)
        pose_handle_world_init = p.getLinkState(bb.bb_id, mech.handle_id)[:2]

        # calculate trajectory
        pose_handle_base_world = mech.get_pose_handle_base_world()
        traj = policy.generate_trajectory(pose_handle_base_world, config_goal, debug)

        # execute trajectory
        waypoints_reached, duration, joint_motion, pose_handle_world_final = \
                gripper.execute_trajectory(traj, mech, policy.type, random_policy, debug=debug)

        # save result data
        control_params = util.ControlParams(gripper.k, gripper.d, gripper.add_dist, gripper.p_err_thresh, policy.p_delta)
        policy_params = policy.get_policy_tuple()
        mechanism_params = mech.get_mechanism_tuple()
        results += [util.Result(control_params, policy_params, mechanism_params, waypoints_reached,\
                    joint_motion, pose_handle_world_init, pose_handle_world_final, config_goal, image_data, git_hash)]

    p.disconnect()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-mech', type=int, default=6)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    test_policy(args.viz, args.debug, args.max_mech, args.random)
    print('done testing policy')
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting...')
