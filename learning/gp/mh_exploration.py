from argparse import Namespace
from gen.generate_policy_data import generate_dataset
from gen.generator_busybox import BusyBox
from actions.policies import generate_policy, Revolute
from utils.setup_pybullet import setup_env
from scipy.stats.distributions import norm
import numpy as np
import pickle
from utils import util
import time

def cost(policy, q, mech, gripper):
    """ Calculate the cost by measuring the distance the mechanism moves. """
    gripper.reset(mech)
    pose_handle_base_world = mech.get_pose_handle_base_world()

    traj = policy.generate_trajectory(pose_handle_base_world, q, debug=False)
    c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False, show=False)

    if motion > 0.05 and q > -0.01:
        print('YIKES', q, motion/mech.get_max_dist())
    return np.exp(motion/0.0175), motion # 0.00175


def proposal(policy, q, bb, s_roll=1., s_pitch=1., s_rad=0.04, s_q=1.):
    params = policy.get_policy_tuple().params
    roll, pitch, radius = params.rot_axis_roll, params.rot_axis_pitch, params.rot_radius_x

    # Sample new parameters according to our proposal distribution.
    new_roll = norm.rvs(loc=roll, scale=s_roll)
    new_pitch = norm.rvs(loc=pitch, scale=s_pitch)
    new_radius = norm.rvs(loc=radius, scale=s_rad)
    new_q = norm.rvs(loc=q, scale=s_q)

    #rev_prob = norm.pdf(roll, loc=new_roll )
    if new_roll > 2*np.pi:
        new_roll -= 2*np.pi
    elif new_roll < 0:
        new_roll += 2*np.pi

    if new_pitch > 2*np.pi:
        new_pitch -= 2*np.pi
    elif new_pitch < 0:
        new_pitch += 2*np.pi

    if new_q < -np.pi/2:
        new_q = -np.pi/2
    elif new_q > 0:
        new_q = 0

    if new_radius < 0.06:
        new_radius = 0.06
    elif new_radius > 0.15:
        new_radius = 0.15

    rot_axis_world = util.quaternion_from_euler(new_roll, new_pitch, 0.0)
    radius = [-new_radius, 0.0, 0.0]
    p_handle_base_world = bb._mechanisms[0].get_pose_handle_base_world().p
    p_rot_center_world = p_handle_base_world + util.transformation(radius, [0., 0., 0.], rot_axis_world)
    rot_orn = [0., 0., 0., 1.]
    new_policy = Revolute(p_rot_center_world,
                          new_roll,
                          new_pitch,
                          rot_axis_world,
                          new_radius,
                          rot_orn)

    return new_policy, new_q, 1, 1


def mh_exploration(bb_result, n_iters):
    bb = BusyBox.bb_from_result(bb_result, urdf_num=0)
    mech = bb._mechanisms[0]
    image_data, gripper = setup_env(bb, False, False, True)

    samples = []

    policy = generate_policy(bb=bb,
                             mech=bb._mechanisms[0],
                             random_policies=False,
                             randomness=1.0)
    q = policy.generate_config(mech=bb._mechanisms[0],
                               goal_config=None)

    for _ in range(n_iters):
        new_policy, new_q, rev_trans_cost, trans_cost = proposal(policy, q, bb, s_q=0.2, s_pitch=0.3, s_roll=0.3, s_rad=0.02)

        c, motion = cost(policy, q, mech, gripper)
        gripper.reset(mech)
        new_c, _ = cost(new_policy, new_q, mech, gripper)
        gripper.reset(mech)
        print(motion / mech.get_max_dist())

        acc = new_c*rev_trans_cost/(c*trans_cost)
        r = np.min([acc, 1])

        samples.append(util.Result(policy.get_policy_tuple(),
                                   mech.get_mechanism_tuple(),
                                   motion, 0, None, None, q, None, None, 1.0, False))

        if np.random.uniform() < r:
            policy, q = new_policy, new_q

    return samples[0::10]


if __name__ == '__main__':
    # Generate BusyBox.
    bb_dataset_args = Namespace(max_mech=1,
                                mech_types=['door'],
                                urdf_num=0,
                                debug=False,
                                n_bbs=4,
                                n_samples=1,
                                viz=False,
                                random_policies=False,
                                randomness=1.0,
                                goal_config=None,
                                bb_fname=None,
                                no_gripper=True)
    busybox_data = generate_dataset(bb_dataset_args, None)

    samples = []
    for bb in busybox_data:
        samples.append(mh_exploration(bb[0], 1000))

    with open('mh_samples_lin.pickle', 'wb') as handle:
        pickle.dump(samples, handle)