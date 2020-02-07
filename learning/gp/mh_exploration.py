from argparse import Namespace
from gen.generate_policy_data import generate_dataset
from gen.generator_busybox import BusyBox
from actions.policies import generate_policy, Revolute
from utils.setup_pybullet import setup_env
from scipy.stats.distributions import norm
from learning.gp.explore_single_bb import get_nn_preds
import numpy as np
import pickle
from utils import util
import time


def create_single_bb_mh_dataset(bb_result, n_interactions, nn_fname, n_chains=10):
    bb = BusyBox.bb_from_result(bb_result, urdf_num=0)
    mech = bb._mechanisms[0]
    image_data, gripper = setup_env(bb, False, False, True)

    policies = []
    # TODO: If nn_fname is empty, just collect random data.
    if nn_fname == '':
        for _ in range(0, n_interactions):
            policy = generate_policy(bb=bb,
                                     mech=mech,
                                     random_policies=False,
                                     randomness=1.0)
            q = policy.generate_config(mech=bb._mechanisms[0],
                                       goal_config=None)
            policies.append((policy, q))
    else:
        nn = util.load_model(nn_fname, 16, use_cuda=False)
        def nn_callback(p, q, m, g):
            res = util.Result(p.get_policy_tuple(), m.get_mechanism_tuple(), 0, 0, None,
                              None, float(q), image_data, None, 1.0, True)
            preds = get_nn_preds([res], nn, ret_dataset=False, use_cuda=False)
            preds = preds[0]
            return np.exp(preds/0.0125), preds

        # TODO: Otherwise, run a bunch of MCMC chains on the NN.

        print('Start MCMC')
        policies = mh_exploration(bb_result, n_iters=2000, cost_callback=nn_callback)
        print('End MCMC')

        # TODO: Sample to chains so there are n_interactions policies to evaluate.


    # Interact with the BusyBox at the chosen policies.
    dataset = []
    regrets = []
    for policy, q in policies:
        _, motion = true_cost(policy, q, mech, gripper)
        dataset.append(util.Result(policy.get_policy_tuple(),
                                   mech.get_mechanism_tuple(),
                                   float(motion), 0, None, None, float(q), image_data, None, 1.0, True))
        regrets.append(1 - motion/mech.get_max_dist())
    return dataset, None, np.mean(regrets)


def true_cost(policy, q, mech, gripper):
    """ Calculate the cost by measuring the distance the mechanism moves. """
    gripper.reset(mech)
    pose_handle_base_world = mech.get_pose_handle_base_world()

    traj = policy.generate_trajectory(pose_handle_base_world, q, debug=False)
    c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False, show=False)

    if motion > 0.05 and q > -0.01:
        print('YIKES', q, motion/mech.get_max_dist())
    return np.exp(motion/0.0075), motion  # 0.0175


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


def mh_exploration(bb_result, n_iters, cost_callback):
    bb = BusyBox.bb_from_result(bb_result, urdf_num=0)
    mech = bb._mechanisms[0]
    image_data, gripper = setup_env(bb, False, False, True)

    samples = []
    policies = []
    policy = generate_policy(bb=bb,
                             mech=bb._mechanisms[0],
                             random_policies=False,
                             randomness=1.0)
    q = policy.generate_config(mech=bb._mechanisms[0],
                               goal_config=None)

    for _ in range(n_iters):
        # new_policy, new_q, rev_trans_cost, trans_cost = proposal(policy, q, bb, s_q=0.2, s_pitch=0.3, s_roll=0.3, s_rad=0.02)
        new_policy, new_q, rev_trans_cost, trans_cost = proposal(policy, q, bb, s_q=0.2, s_pitch=0.1, s_roll=0.1, s_rad=0.01)
        c, motion = cost_callback(policy, q, mech, gripper)
        gripper.reset(mech)
        new_c, _ = cost_callback(new_policy, new_q, mech, gripper)
        gripper.reset(mech)
        print(motion / mech.get_max_dist())

        acc = new_c*rev_trans_cost/(c*trans_cost)
        r = np.min([acc, 1])
        policies.append((policy, q))
        samples.append(util.Result(policy.get_policy_tuple(),
                                   mech.get_mechanism_tuple(),
                                   motion, 0, None, None, q, image_data, None, 1.0, False))

        if np.random.uniform() < r:
            policy, q = new_policy, new_q
    return samples[::10]
    return policies[1000::10]  # , samples[0::10]


if __name__ == '__main__':
    # Generate BusyBox.
    bb_dataset_args = Namespace(max_mech=1,
                                mech_types=['door'],
                                urdf_num=0,
                                debug=False,
                                n_bbs=1,
                                n_samples=1,
                                viz=False,
                                random_policies=False,
                                randomness=1.0,
                                goal_config=None,
                                bb_fname=None,
                                no_gripper=True)
    busybox_data = generate_dataset(bb_dataset_args, None)
    print(busybox_data[0][0].mechanism_params)
    input()
    n_chains = 3
    for cx in range(n_chains):
        samples = []
        for bb in busybox_data:
            samples.append(mh_exploration(bb[0], 2000, cost_callback=true_cost))

        with open('mh_samples_true9_0075_chain_%d.pickle' % cx, 'wb') as handle:
            pickle.dump(samples, handle)