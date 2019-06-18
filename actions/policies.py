from collections import namedtuple
import numpy as np
from util import util
import itertools
import pybullet as p

"""See actions.gripper.py for pose, p, and q naming conventions and variable names
"""
class Policy(object):
    def __init__(self, type, p_delta=None):
        """ This is an interface for each Policy type
        :param type: string, name of the child Policy class
        :param p_delta (optional): the distance between trajectory waypoints
        """
        self.type = type
        self.p_delta = 0.01 if p_delta is None else p_delta

    def generate_trajectory(self, pose_tip_world_init, p_joint_world_init, config_goal, debug=False):
        """ This method generates a trajectory of waypoints that the gripper tip should
        move through
        :param pose_tip_world_init: util.Pose, initial grasp pose of the gripper tip
        :param p_joint_world_init: a vector of length 3, the initial (x,y,z) position
                                    of the joint
        :param config_goal: the goal configuration of the joint
        :param debug (optional): if True, display debig visualizations
        """
        q_joint_world = np.array([0.0, 0.0, 0.0, 1.0])
        config_curr = self.inverse_kinematics(p_joint_world_init, q_joint_world)
        config_dir_unit = config_dir(config_curr, config_goal)
        config_delta = self._config_delta_mag*config_dir_unit

        # initial offset between joint pose and gripper pose
        # assume that we want to keep this offset constant throughout the traj
        q_tip_joint = util.quat_math(q_joint_world, pose_tip_world_init.q, True, False)
        p_tip_joint = util.transformation(pose_tip_world_init.p, p_joint_world_init, joint_orn, inverse=True)

        poses = []
        for i in itertools.count():
            if past_goal_config(config_curr, config_goal, config_dir_unit):
                break
            pose_joint_world = self.forward_kinematics(config_curr)
            q_tip_world = util.quat_math(pose_joint_world.q, q_tip_joint, False, False)
            p_tip_world = util.transformation(p_tip_joint, *pose_joint_world)
            poses += [util.Pose(p_tip_world, q_tip_world)]
            config_curr += config_delta
            if debug:
                if i>0:
                    p.addUserDebugLine(poses[i-1].p, poses[i].p)
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
    def __init__(self, pos, orn, dir, p_delta):
        """
        :param pos: vector of length 3, a rigid (x,y,z) position in the world frame
                    along the prismatic joint
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion in the world
                    frame representing the orientation of the handle in the world
        :param dir: unit vector of length 3, the direction of the prismatic joint
                    in the world frame
        :param p_delta: scalar, the distance between waypoints in the generated
                        trajectories
        """
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.prismatic_dir = dir

        # derived
        self._origin_M = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)

        super(Prismatic,self).__init__('Prismatic', p_delta)
        self._config_delta_mag = self.p_delta

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
    def model(bb, mech, p_delta=None):
        p_track_w = p.getLinkState(bb.bb_id,mech.track_id)[0]
        rigid_position = bb.project_onto_backboard(p_track_w)
        rigid_orientation = [0., 0., 0., 1.]
        prismatic_dir = [mech.axis[0], 0., mech.axis[1]]
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir, p_delta)

    @staticmethod
    def random(bb, p_delta=None):
        rigid_position = random_pos(bb)
        rigid_orientation = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        angle = np.random.uniform(0, np.pi)
        prismatic_dir = np.array([np.cos(angle), 0., np.sin(angle)])
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir, p_delta)

class Revolute(Policy):
    def __init__(self, center, axis, radius, orn, p_delta):
        self.rot_center = center
        self.rot_axis = axis
        self.rot_radius = radius
        self.rot_orientation = orn

        # derived
        self.center = util.pose_to_matrix(self.rot_center, self.rot_axis)
        self.radius = util.pose_to_matrix(self.rot_radius, self.rot_orientation)

        super(Revolute,self).__init__('Revolute', p_delta)
        self._config_delta_mag = self.p_delta/np.linalg.norm(self.radius)

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
    def model(bb, mech, p_delta):
        p_door_base_w = p.getLinkState(bb.bb_id, mech.door_base_id)[0]
        p_handle_w = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        rot_center = bb.project_onto_backboard([p_door_base_w[0], p_door_base_w[1], p_handle_w[2]])
        rot_radius = np.subtract([p_handle_w[0],rot_center[1],p_handle_w[2]],rot_center)
        rot_axis = [0., 0., 0., 1.]
        rot_orientation = [0., 0., 0., 1.]
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation, p_delta)

    @staticmethod
    def random(bb, p_delta):
        rot_center = random_pos(bb)
        rot_axis = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        rot_radius = random_radius()
        rot_orientation = np.array([0.,0.,0.,1.]) # this is hard coded in the joint models as well
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation, p_delta)

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
def generate_random_policy(bb, p_delta=None, policy_types=[Revolute, Prismatic]):
    policy_type = np.random.choice(policy_types)
    return policy_type.random(bb, p_delta)

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

def config_dir(config_curr, config_goal):
    return 1 if (config_goal > config_curr) else -1

def past_goal_config(curr_q, config_goal, q_dir_unit):
    return (curr_q > config_goal) if (q_dir_unit == 1) else (curr_q < config_goal)
