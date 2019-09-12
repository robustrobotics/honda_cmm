import argparse
import numpy as np
from actions.policies import Prismatic
from gen.generator_busybox import BusyBox, Slider
from util import util
import matplotlib.pyplot as plt
from util.setup_pybullet import setup_env
from actions.gripper import Gripper
import pybullet as p

# ONLY FOR PRISMATIC

def execute_systematic(args):
    fig, ax = plt.subplots()

    if args.bb_file:
        bb_data = util.read_from_file(args.bb_file)
    max_regrets = []
    for n in range(args.N):
        # generate busybox and setup pybullet env
        if args.bb_file:
            bb = BusyBox.bb_from_result(bb_data[n])
        else:
            bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider], urdf_tag=args.urdf_num, debug=args.debug)
        mech = bb._mechanisms[0]
        mech_range = mech.range/2
        setup_env(bb, args.viz, args.debug)
        gripper = Gripper(bb.bb_id)

        # generate list of goals
        delta_pitches = np.linspace(-np.pi/2, np.pi/2, args.T)
        config_goal = 0.25

        # try each goal
        regrets = []
        for dp in delta_pitches:
            # generate policy and execute
            policy = Prismatic._gen(bb, mech, randomness=0, delta_pitch=dp)
            pose_handle_base_world = mech.get_pose_handle_base_world()
            traj = policy.generate_trajectory(pose_handle_base_world, config_goal, args.debug)
            _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, args.debug)

            # calculate regret
            regrets += [(mech_range - motion)/mech_range]

            # reset mechanism and gripper
            p.resetJointState(bb.bb_id, mech.handle_id, 0.0)
            gripper._set_pose_tip_world(gripper.pose_tip_world_reset)
        max_regrets += [min(regrets)]
        print('Busybox', n, 'Max Regret :', max(regrets))
    final_regret = np.mean(max_regrets)
    print('Final Regret: ', final_regret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--N', type=int, default=5) # number bbs to test on
    parser.add_argument('--T', type=int, default=5) # number directions to try per bb
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--bb-file', type=str)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        execute_systematic(args)
    except:
        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
