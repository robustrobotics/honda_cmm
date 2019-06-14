from collections import namedtuple
from joints import JointModel, Prismatic, Revolute
import numpy as np
import util
import itertools
import pybullet as p

class Policy(object):
    def __init__(self, type, delta_pos = .001):
        self.type = type
        # max distance between waypoints in trajectory
        self.delta_pos = delta_pos

    def generate_trajectory(self, debug=False):
        joint_orn = np.array([0., 0., 0., 1.])
        curr_q = self.joint.inverse_kinematics(self.init_joint_pos, joint_orn)
        q_dir_unit = q_dir(curr_q, self.goal_q)
        delta_q = self.delta_q_mag*q_dir_unit

        # initial offset between joint pose orn and gripper orn
        # assume that we want to keep this offset constant throughout the traj
        init_delta_q = util.quat_math(joint_orn, self.init_grasp_orn, True, False)

        poses = []
        for i in itertools.count():
            if near(curr_q, self.goal_q, q_dir_unit):
                break
            curr_joint_pose = self.joint.forward_kinematics(curr_q)
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

class Poke(Policy):
    def __init__(self):
        super(Poke,self).__init__('Poke')

    def generate_trajectory(self):
        pass

    @staticmethod
    def generate_random_params():
        pass

class Slide(Policy):
    def __init__(self):
        super(Slide,self).__init__('Slide')

    def generate_trajectory(self):
        pass

    @staticmethod
    def generate_random_params():
        pass

class Prism(Policy):
    def __init__(self, grasp_orn, joint_pos, pos, orn, dir, goal_q):
        self.init_grasp_orn = grasp_orn
        self.init_joint_pos = joint_pos
        self.pos = pos
        self.orn = orn
        self.dir = dir
        self.goal_q = goal_q

        self.joint = Prismatic(self.pos, self.orn, self.dir)
        super(Prism,self).__init__('Prism')
        self.delta_q_mag = self.delta_pos

    @staticmethod
    def generate_random_params():
        pass

class Rev(Policy):
    def __init__(self, grasp_orn, joint_pos, center, axis, radius, orn, goal_q):
        self.init_grasp_orn = grasp_orn
        self.init_joint_pos = joint_pos
        self.center = center
        self.axis = axis
        self.radius = radius
        self.orn = orn
        self.goal_q = goal_q

        self.joint = Revolute(self.center, self.axis, self.radius, self.orn)
        super(Rev,self).__init__('Rev')
        self.delta_q_mag = self.delta_pos/np.linalg.norm(self.radius)

    @staticmethod
    def generate_random_params():
        pass

class Path(Policy):
    def __init__(self):
        super(Path,self).__init__('Path')

    def generate_trajectory(self):
        pass

    @staticmethod
    def generate_random_params():
        pass

# couldn't make staticmethod of Policy because needed child class types first
def generate_random_policy(policy_types=[Poke, Slide, Rev, Prism, Path]):
    policy = np.random.choice(policy_types)
    params = policy.generate_random_parameters()
    return policy

## Helper Functions
def q_dir(curr_q, goal_q):
    return 1 if (goal_q > curr_q) else -1

def near(curr_q, goal_q, q_dir_unit):
    return (curr_q > goal_q) if (q_dir_unit == 1) else (curr_q < goal_q)
