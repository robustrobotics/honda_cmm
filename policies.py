from collections import namedtuple
import numpy as np
import util
import itertools
import pybullet as p

class Policy(object):
    def __init__(self, type, delta_pos = .001):
        self.type = type
        # max distance between waypoints in trajectory
        self.delta_pos = delta_pos

    def generate_trajectory(self, init_grasp_pose, init_joint_pos, debug=False):
        joint_orn = np.array([0., 0., 0., 1.])
        curr_q = self.inverse_kinematics(init_joint_pos, joint_orn)
        q_dir_unit = q_dir(curr_q, self.goal_q)
        delta_q = self.delta_q_mag*q_dir_unit

        # initial offset between joint pose orn and gripper orn
        # assume that we want to keep this offset constant throughout the traj
        init_delta_q = util.quat_math(joint_orn, init_grasp_pose.orn, True, False)
        init_delta_pos = np.subtract(init_grasp_pose.pos,init_joint_pos)

        poses = []
        for i in itertools.count():
            if near(curr_q, self.goal_q, q_dir_unit):
                break
            curr_joint_pose = self.forward_kinematics(curr_q)
            # for rev, orn should change with q
            # for prism, orn should remain constant
            # for now remain constant for both
            grasp_orn = util.quat_math(curr_joint_pose.orn, init_delta_q, False, False)
            grasp_pos = np.add(curr_joint_pose.pos,init_delta_pos)
            poses += [util.Pose(grasp_pos, grasp_orn)]
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
    def random():
        pass

class Slide(Policy):
    def __init__(self):
        super(Slide,self).__init__('Slide')

    def generate_trajectory(self):
        pass

    @staticmethod
    def random():
        pass

class Prism(Policy):
    def __init__(self, pos, orn, dir, goal_q):
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.prismatic_dir = dir
        self.goal_q = goal_q

        # derived
        self.origin_M = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)

        super(Prism,self).__init__('Prism')
        self.delta_q_mag = self.delta_pos

    def forward_kinematics(self, q):
        q_dir = np.multiply(q, self.prismatic_dir)
        q_dir = np.concatenate([q_dir, [1.]])
        p_z = np.dot(self.origin_M, q_dir)[:3]
        q_z = util.quaternion_from_matrix(self.origin_M)
        return util.Pose(p_z, q_z)

    def inverse_kinematics(self, p_z, q_z):
        z_M = util.pose_to_matrix(p_z, q_z)
        inv_o_M = np.linalg.inv(self.origin_M)
        o_inv_z_M = np.dot(inv_o_M,z_M)
        trans = o_inv_z_M[:3,3]
        return np.dot(self.prismatic_dir, trans)

    @staticmethod
    def random(bb_center, width, height):
        pos = random_pos(bb_center, width, height)
        orn = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models aswell
        dir = random_unit_vector()
        goal_q = random_configuration()
        return Prism(pos, orn, dir, goal_q)

class Rev(Policy):
    def __init__(self, center, axis, radius, orn, goal_q):
        self.rot_center = center
        self.rot_axis = axis
        self.rot_radius = radius
        self.rot_orientation = orn
        self.goal_q = goal_q

        # derived
        self.center = util.pose_to_matrix(self.rot_center, self.rot_axis)
        self.radius = util.pose_to_matrix(self.rot_radius, self.rot_orientation)

        super(Rev,self).__init__('Rev')
        self.delta_q_mag = self.delta_pos/np.linalg.norm(self.radius)

    def forward_kinematics(self, q):
        rot_z = util.trans.rotation_matrix(-q,[0,0,1])
        M = util.trans.concatenate_matrices(self.center,rot_z,self.radius)
        p_z = M[:3,3]
        q_z = util.quaternion_from_matrix(M)
        return util.Pose(p_z, q_z)

    def inverse_kinematics(self, p_z, q_z):
        z = util.pose_to_matrix(p_z, q_z)
        z_inv_c = np.dot(np.linalg.inv(z),self.center)
        inv_r = np.dot(np.linalg.inv(self.radius),z_inv_c)
        angle, direction, point = util.trans.rotation_from_matrix(inv_r)
        return angle

    @staticmethod
    def random(bb_center, width, height):
        center = random_pos(bb_center, width, height)
        axis = np.array(0.,0.,0.,1.)
        radius = random_radius(bb_center, width, height)
        orn = np.array(0.,0.,0.,1.)
        goal_q = random_angle()
        return Rev(center, axis, radius, orn, goal_q)

class Path(Policy):
    def __init__(self):
        super(Path,self).__init__('Path')

    def generate_trajectory(self):
        pass

    @staticmethod
    def random():
        pass

# couldn't make staticmethod of Policy because needed child class types first
# TODO: add back other policy types as they're made
def generate_random_policy(bb_center, width, height, policy_types=[Rev, Prism]):
    policy_type = np.random.choice(policy_types)
    return policy_type.random(bb_center, width, height)

## Helper Functions
def random_pos(bb_center, width, height):
    x_limits = np.add(bb_center[0], [-width/2,width/2])
    y_limits = np.random.uniform([.23,.27])
    z_limits = np.add(bb_center[2], [-height/2,height/2])
    x = np.random.uniform(*x_limits)
    y = np.random.uniform(*y_limits)
    z = np.random.uniform(*z_limits)
    return (x, y, z)

def random_orn():
    orn_M = util.trans.random_rotation_matrix()
    orn_q = util.quaternion_from_matrix(orn_M)
    return orn_q

def random_unit_vector():
    orn_M = util.trans.random_rotation_matrix()
    unit_vector = np.multiply(orn_M,[1,0,0])
    return unit_vector

def random_configuration():
    return np.random.uniform(-10.,10.)

def q_dir(curr_q, goal_q):
    return 1 if (goal_q > curr_q) else -1

def near(curr_q, goal_q, q_dir_unit):
    return (curr_q > goal_q) if (q_dir_unit == 1) else (curr_q < goal_q)
