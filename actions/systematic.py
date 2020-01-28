import argparse
import numpy as np
from actions.policies import Prismatic
from gen.generator_busybox import BusyBox, Slider
from utils import util
from utils.setup_pybullet import setup_env
import pybullet as p

# ONLY FOR PRISMATIC

def calc_systematic_policies(pos, orn, T):
    # generate list of goals
    pitches = np.linspace(-np.pi, 0.0, T)
    policies = []
    for pitch in pitches:
        policies += [Prismatic(pos, orn, pitch, 0.0)]
    return policies

def execute_systematic(args):
    if args.bb_fname:
        bb_data = util.read_from_file(args.bb_fname)

    avg_regrets = []
    min_regrets = []

    for n in range(args.N):
        # generate busybox and setup pybullet env
        if args.bb_fname:
            bb = BusyBox.bb_from_result(bb_data[n])
        else:
            bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider], urdf_tag=args.urdf_num, debug=args.debug)
        mech = bb._mechanisms[0]
        mech_range = mech.range/2
        _, gripper = setup_env(bb, args.viz, args.debug, args.no_gripper)
        pos = mech.get_pose_handle_base_world().p
        orn = [0., 0., 0., 1.]
        policies = calc_systematic_policies(pos, orn, args.T)

        # try each goal
        regrets = []
        for policy in policies:
            # generate policy and execute
            pose_handle_base_world = mech.get_pose_handle_base_world()
            traj = policy.generate_trajectory(pose_handle_base_world, args.debug)
            _, motion, _ = gripper.execute_trajectory(traj, mech, policy.type, args.debug)

            # calculate regret
            regrets += [(mech_range - motion)/mech_range]

            # reset mechanism and gripper
            gripper.reset(mech)
            gripper._set_pose_tip_world(gripper.pose_tip_world_reset)
        avg_regrets += [np.mean(regrets)]
        min_regrets += [min(regrets)]
        #print('Busybox', n, 'Min Regret :', min(regrets))
    final_regret = np.mean(min_regrets)
    results = {'T': args.T, 'final': final_regret, 'avg_regrets': avg_regrets, 'min_regrets': min_regrets}
    print('Final Regret:', final_regret)
    fname = args.fname
    if not fname:
        fname = 'systematic_n'+str(args.N)+'_t'+str(args.T)+'.pickle'
    util.write_to_file(fname, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--N', type=int, default=5) # number bbs to test on
    parser.add_argument('--T', type=int, default=5) # number directions to try per bb
    # if running multiple gens, give then a urdf_num so the correct urdf is read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    parser.add_argument('--bb-fname', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--no-gripper', action='store_true')
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
