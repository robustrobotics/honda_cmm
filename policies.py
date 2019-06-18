from collections import namedtuple
import numpy as np
import util
import itertools
import pybullet as p

class Policy(object):
    def __init__(self, type, delta_pos = .01):
        self.type = type
        # max distance between waypoints in trajectory
        self.delta_pos = delta_pos

    def generate_trajectory(self, init_grasp_pose, init_joint_pos, goal_q, debug=False):
        joint_orn = np.array([0., 0., 0., 1.])
        curr_q = self.inverse_kinematics(init_joint_pos, joint_orn)
        q_dir_unit = q_dir(curr_q, goal_q)
        delta_q = self.delta_q_mag*q_dir_unit

        # initial offset between joint pose orn and gripper orn
        # assume that we want to keep this offset constant throughout the traj
        init_delta_q = util.quat_math(joint_orn, init_grasp_pose.orn, True, False)
        pos_grasp_joint = util.transformation(init_grasp_pose.pos, init_joint_pos, joint_orn, inverse=True)


        poses = []
        for i in itertools.count():
            if near(curr_q, goal_q, q_dir_unit):
                break
            curr_joint_pose = self.forward_kinematics(curr_q)
            # for rev, orn should change with q
            # for prism, orn should remain constant
            # for now remain constant for both
            grasp_orn = util.quat_math(curr_joint_pose.orn, init_delta_q, False, False)

            pos_grasp_world = util.transformation(pos_grasp_joint, *curr_joint_pose)
            poses += [util.Pose(pos_grasp_world, grasp_orn)]
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

class Prismatic(Policy):
    def __init__(self, pos, orn, dir):
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.prismatic_dir = dir

        # derived
        self.origin_M = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)

        super(Prismatic,self).__init__('Prismatic')
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

    def generate_random_config(self):
        return np.random.uniform(-0.5,0.5)

    @staticmethod
    def model(bb, mech):
        p_track_w = p.getLinkState(bb.bb_id,mech.track_id)[0]
        rigid_position = bb.project_onto_backboard(p_track_w)
        rigid_orientation = [0., 0., 0., 1.]
        prismatic_dir = [mech.axis[0], 0., mech.axis[1]]
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir)

    @staticmethod
    def random(bb):
        rigid_position = random_pos(bb)
        rigid_orientation = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        angle = np.random.uniform(0, np.pi)
        prismatic_dir = np.array([np.cos(angle), 0., np.sin(angle)])
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir)

class Revolute(Policy):
    def __init__(self, center, axis, radius, orn):
        self.rot_center = center
        self.rot_axis = axis
        self.rot_radius = radius
        self.rot_orientation = orn

        # derived
        self.center = util.pose_to_matrix(self.rot_center, self.rot_axis)
        self.radius = util.pose_to_matrix(self.rot_radius, self.rot_orientation)

        super(Revolute,self).__init__('Revolute')
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

    def generate_random_config(self):
        return np.random.uniform(-2*np.pi,2*np.pi)

    @staticmethod
    def model(bb, mech):
        p_door_base_w = p.getLinkState(bb.bb_id, mech.door_base_id)[0]
        p_handle_w = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        rot_center = bb.project_onto_backboard([p_door_base_w[0], p_door_base_w[1], p_handle_w[2]])
        rot_radius = np.subtract([p_handle_w[0],rot_center[1],p_handle_w[2]],rot_center)
        rot_axis = [0., 0., 0., 1.]
        rot_orientation = [0., 0., 0., 1.]
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation)

    @staticmethod
    def random(bb):
        rot_center = random_pos(bb)
        rot_axis = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        rot_radius = random_radius()
        rot_orientation = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation)

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
def generate_random_policy(bb, policy_types=[Revolute, Prismatic]):
    policy_type = np.random.choice(policy_types)
    return policy_type.random(bb)

## Helper Functions
def random_pos(bb):
    bb_center = p.getLinkState(bb.bb_id,0)[0]
    x_limits = np.add(bb_center[0], [-bb.width/2,bb.width/2])
    z_limits = np.add(bb_center[2], [-bb.height/2,bb.height/2])
    x = np.random.uniform(*x_limits)
    y = bb.project_onto_backboard([0., 0., 0.,])[1]
    z = np.random.uniform(*z_limits)
    return (x, y, z)

def random_radius():
    r = np.random.uniform(0.05,0.15)
    return np.array([r, 0., 0.])

def q_dir(curr_q, goal_q):
    return 1 if (goal_q > curr_q) else -1

def near(curr_q, goal_q, q_dir_unit):
    return (curr_q > goal_q) if (q_dir_unit == 1) else (curr_q < goal_q)
