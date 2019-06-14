from collections import namedtuple
from joints import JointModel, Prismatic, Revolute
import numpy as np
import util
import itertools
import pybullet as p

# named tuples for all policy params
#PokeParams = namedtuple('PokeParams', '...')
#SlideParams = namedtuple('SlideParams', '...')
PrismParams = namedtuple('PrismParams', 'grasp_orn joint_pos pos orn dir goal_q')
RevParams = namedtuple('RevParams', 'grasp_orn joint_pos center axis radius orn goal_q')
#PathParams = namedtuple('PathParams', '...')

# max distance between waypoints in trajectory
delta_pos = .001

# all primitive parameterized policies take in parameters and output trajectories
# of poses for the gripper to follow
def poke(params, debug=False):
    pass

def slide(params, debug=False):
    pass

def prism(params, debug=False):
    delta_q_mag = delta_pos
    return from_model('prismatic', params, delta_q_mag=delta_q_mag, debug=debug)

def rev(params, debug=False):
    delta_q_mag = delta_pos/np.linalg.norm(params.radius)
    return from_model('revolute', params, delta_q_mag=delta_q_mag, debug=debug)

def path(params, debug=False):
    pass

## Helper Functions
def from_model(model_type, params, delta_q_mag, debug=False):
    if model_type == 'prismatic':
        joint = Prismatic(params.pos, params.orn, params.dir)
    else:
        joint = Revolute(params.center, params.axis, params.radius, params.orn)

    joint_orn = np.array([0., 0., 0., 1.])
    curr_q = joint.inverse_kinematics(params.joint_pos, joint_orn)
    q_dir_unit = q_dir(curr_q, params.goal_q)
    delta_q = delta_q_mag*q_dir_unit

    # initial offset between joint pose orn and gripper orn
    # assume that we want to keep this offset constant throughout the traj
    init_delta_q = util.quat_math(joint_orn, params.grasp_orn, True, False)

    poses = []
    for i in itertools.count():
        if near(curr_q, params.goal_q, q_dir_unit):
            break
        curr_joint_pose = joint.forward_kinematics(curr_q)
        # for rev, orn should change with q
        # for prism, orn should remain constant
        # for now remain constant for both
        grasp_orn = util.quat_math(curr_joint_pose.orn, init_delta_q, False, False)
        poses += [util.Pose(curr_joint_pose.pos, grasp_orn)]
        curr_q += delta_q
        if debug:
            if i>0:
                p.addUserDebugLine(poses[i-1].pos, poses[i].pos)
    return poses

def q_dir(curr_q, goal_q):
    return 1 if (goal_q > curr_q) else -1

def near(curr_q, goal_q, q_dir_unit):
    return (curr_q > goal_q) if (q_dir_unit == 1) else (curr_q < goal_q)
