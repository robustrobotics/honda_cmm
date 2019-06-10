from collections import namedtuple
from joints import JointModel, Prismatic, Revolute
import numpy as np
import util

# named tuples for all policy params
#PokeParams = namedtuple('PokeParams', '...')
#SlideParams = namedtuple('SlideParams', '...')
PrismParams = namedtuple('PrismParams', 'grasp_pose pos orn dir goal_q')
RevParams = namedtuple('RevParams', 'grasp_pose center axis radius orn goal_q')
#PathParams = namedtuple('PathParams', '...')

# all primitive parameterized policies take in parameters and output trajectories
# of poses for the gripper to follow
def poke(params):
    pass

def slide(params):
    pass

def prism(params):
    delta_q_mag = .01
    return from_model('prismatic', params, delta_q_mag=delta_q_mag)

def rev(params):
    delta_q_mag = np.pi/100
    return from_model('revolute', params, delta_q_mag=delta_q_mag)

def path(params):
    pass

## Helper Functions
def from_model(model_type, params, delta_q_mag):
    if model_type == 'prismatic':
        joint = Prismatic(params.pos, params.orn, params.dir)
    else:
        joint = Revolute(params.center, params.axis, params.radius, params.orn)

    start_pose = params.grasp_pose
    curr_q = joint.inverse_kinematics(start_pose.pos, start_pose.orn)
    q_dir_unit = q_dir(curr_q, params.goal_q)
    delta_q = delta_q_mag*q_dir_unit

    poses = [start_pose]
    while not near(curr_q, params.goal_q, q_dir_unit):
        curr_q += delta_q
        curr_joint_pose = joint.forward_kinematics(curr_q)
        # for rev, orn should change with q
        # for prism, orn should remain constant
        # for now remain constant for both
        poses += [util.Pose(curr_joint_pose.pos, start_pose.orn)]
    print(len(poses))
    return poses

def q_dir(curr_q, goal_q):
    return 1 if (goal_q > curr_q) else 0

def near(curr_q, goal_q, q_dir_unit):
    return (curr_q > goal_q) if (q_dir_unit == 1) else (curr_q < goal_q)
