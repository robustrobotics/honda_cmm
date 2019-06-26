from collections import namedtuple
import numpy as np
from util import util
import itertools
import pybullet as p

"""See actions.gripper for variable naming and naming conventions
"""

PolicyParams = namedtuple('PolicyParams', 'type params')
"""
Tuple for storing policy data
:param type: str, name of the Policy object type
:param params: one of {actions.policies.PrismaticParams, actions.policies.RevoluteParams}
"""

PrismaticParams = namedtuple('PrismaticParams', 'rigid_position rigid_orientation prismatic_dir')

RevoluteParams = namedtuple('RevoluteParams', 'rot_center rot_axis rot_radius rot_orientation')

class Policy(object):
    def __init__(self, type, p_delta=None):
        """ This is an interface for each Policy type. Each Policy must implement the
        generate_trajectory, _forward_kinematics, _inverse_kinematics, generate_random_config,
        _random, and _model methods
        :param type: string, name of the child Policy class
        :param p_delta (optional): scalar, the distance between trajectory waypoints
        """
        self.type = type
        self.p_delta = 0.01 if p_delta is None else p_delta

    def generate_trajectory(self, p_handle_base_world, config_goal, debug=False):
        """ This method generates a trajectory of waypoints that the gripper tip should
        move through
        :param p_handle_base_world_init: util.Pose, initial pose of the base of the handle
        :param config_goal: the goal configuration of the joint
        :param debug (optional): if True, display debug visualizations
        """
        # TODO: don't assume handle always starts at config = 0
        config_curr = self._inverse_kinematics(p_handle_base_world, [0.0, 0.0, 0.0, 1.0])
        config_dir_unit = self._config_dir(config_curr, config_goal)
        config_delta = self._config_delta_mag*config_dir_unit

        poses = []
        for i in itertools.count():
            if self._past_goal_config(config_curr, config_goal, config_dir_unit):
                break
            pose_handle_base_world = self._forward_kinematics(config_curr)
            poses += [pose_handle_base_world]
            config_curr += config_delta
            if debug:
                # draws the planned handle base trajectory
                if i>0:
                    p.addUserDebugLine(poses[i-1].p, poses[i].p)
        return poses

    @staticmethod
    def generate_model_based_config(mech, random=False):
        raise NotImplementedError('generate_model_based_config not implemented for policy type '+self.type)

    def generate_random_config(self):
        raise NotImplementedError('generate_random_config not implemented for policy type '+self.type)

    def get_params_tuple(self):
        raise NotImplementedError('get_params_tuple not implemented for policy type '+self.type)

    def _config_dir(self, config_curr, config_goal):
        return 1 if (config_goal > config_curr) else -1

    def _past_goal_config(self, curr_q, config_goal, q_dir_unit):
        return (curr_q > config_goal) if (q_dir_unit == 1) else (curr_q < config_goal)

    def _forward_kinematics(self, config):
        raise NotImplementedError('_forward_kinematics function not implemented for policy type '+self.type)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        raise NotImplementedError('_inverse_kinematics function not implemented for policy type '+self.type)

    @staticmethod
    def _random(bb, p_delta=None):
        raise NotImplementedError('_random not implemented for policy type '+self.type)

    @staticmethod
    def _model(bb, mech, p_delta=None):
        raise NotImplementedError('_model not implemented for policy type '+self.type)

class Prismatic(Policy):
    def __init__(self, pos, orn, dir, p_delta=None):
        """
        :param pos: vector of length 3, a rigid (x,y,z) position in the world frame
                    along the prismatic joint
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion in the world
                    frame representing the orientation of the handle in the world
        :param dir: unit vector of length 3, the direction of the prismatic joint
                    in the world frame
        :param p_delta (optional): scalar, the distance between waypoints in the generated
                        trajectories
        """
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.prismatic_dir = dir

        # derived
        self._M_origin_world = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)
        super(Prismatic,self).__init__('Prismatic', p_delta)
        self._config_delta_mag = self.p_delta

    def _forward_kinematics(self, config):
        p_joint_origin = np.multiply(config, self.prismatic_dir)
        p_joint_origin_4 = np.concatenate([p_joint_origin, [1.]])
        p_joint_world = np.dot(self._M_origin_world, p_joint_origin_4)[:3]
        q_joint_world = util.quaternion_from_matrix(self._M_origin_world)
        return util.Pose(p_joint_world, q_joint_world)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        M_joint_world = util.pose_to_matrix(p_joint_world, q_joint_world)
        M_world_origin = np.linalg.inv(self._M_origin_world)
        M_joint_origin = np.dot(M_world_origin, M_joint_world)
        p_joint_origin = M_joint_origin[:3,3]
        return np.dot(self.prismatic_dir, p_joint_origin)

    @staticmethod
    def generate_model_based_config(mech, random=False):
        if random:
            return np.random.uniform(-mech.range/2.0, mech.range/2.0)
        else:
            return np.random.choice([-mech.range/2.0, mech.range/2.0])

    def generate_random_config(self):
        """ This function generates a random prismatic joint configuration. The range is
        based on the data.generator range of random prismatic joint track lengths
        """
        return np.random.uniform(-0.5,0.5)

    def get_params_tuple(self):
        prim_params = PrismaticParams(self.rigid_position, self.rigid_orientation, self.prismatic_dir)
        return PolicyParams(self.type, prim_params)

    @staticmethod
    def _model(bb, mech, p_delta=None):
        """ This function generates a Prismatic policy from the mechanism model
        """
        p_track_world = p.getLinkState(bb.bb_id,mech.track_id)[0]
        rigid_position = bb.project_onto_backboard(p_track_world)
        rigid_orientation = [0., 0., 0., 1.]
        prismatic_dir = [mech.axis[0], 0., mech.axis[1]]
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir, p_delta)

    @staticmethod
    def _random(bb, p_delta=None):
        """ This function generates a random Prismatic policy. The ranges are
        based on the data.generator range prismatic joints
        """
        rigid_position = _random_p(bb)
        rigid_orientation = np.array([0.,0.,0.,1.])
        angle = np.random.uniform(0, np.pi)
        prismatic_dir = np.array([np.cos(angle), 0., np.sin(angle)])
        return Prismatic(rigid_position, rigid_orientation, prismatic_dir, p_delta)

class Revolute(Policy):
    def __init__(self, center, axis, radius, orn, p_delta=None):
        """
        :param center: vector of length 3, a rigid (x,y,z) position in the world frame
                    of the center of rotation
        :param axis: vector of length 4, a rigid (x,y,z,w) quaternion in the world
                    frame representing the orientation of the center of rotation
        :param radius: vector of length 3, an (x,y,z) position of the radius/handle base
                        in the util.Pose(center,axis) frame
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion representing the
                    rotataion from axis to the handle base frame
        :param p_delta (optional): scalar, the distance between waypoints in the generated
                        trajectories
        """
        self.rot_center = center
        self.rot_axis = axis
        self.rot_radius = radius
        self.rot_orientation = orn

        # derived
        self._M_center_world = util.pose_to_matrix(self.rot_center, self.rot_axis)
        self._M_radius_center = util.pose_to_matrix(self.rot_radius, self.rot_orientation)
        super(Revolute,self).__init__('Revolute', p_delta)
        self._config_delta_mag = self.p_delta/np.linalg.norm(self.rot_radius)

    def _forward_kinematics(self, config):
        # rotation matrix for a rotation about the z-axis by config radians
        M_joint_z = util.trans.rotation_matrix(-config,[0,0,1])
        M_joint_world = util.trans.concatenate_matrices(self._M_center_world,M_joint_z,self._M_radius_center)
        p_joint_world = M_joint_world[:3,3]
        q_joint_world = util.quaternion_from_matrix(M_joint_world)
        return util.Pose(p_joint_world, q_joint_world)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        M_joint_world = util.pose_to_matrix(p_joint_world, q_joint_world)
        M_joint_center = np.dot(np.linalg.inv(M_joint_world),self._M_center_world)
        # transformation from the radius in the center to the joint in the center
        M_radius_joint_center = np.dot(np.linalg.inv(self._M_radius_center),M_joint_center)
        angle, direction, point = util.trans.rotation_from_matrix(M_radius_joint_center)
        return angle

    @staticmethod
    def generate_model_based_config(mech, random=False):
        if random:
            if mech.flipped:
                return np.random.uniform(0., -np.pi/2.0)
            else:
                return np.random.uniform(0., np.pi/2.0)
        else:
            if mech.flipped:
                return -np.pi/2.0
            else:
                return np.pi/2.0

    def generate_random_config(self):
        """ This function generates a random revolute joint configuration
        """
        return np.random.uniform(-2*np.pi,2*np.pi)

    def get_params_tuple(self):
        prim_params =  RevoluteParams(self.rot_center, self.rot_axis, self.rot_radius, self.rot_orientation)
        return PolicyParams(self.type, prim_params)

    @staticmethod
    def _model(bb, mech, p_delta=None):
        """ This function generates a Revolute policy from the mechanism model
        """
        p_door_world = p.getLinkState(bb.bb_id, mech.door_base_id)[0]
        p_handle_world = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        rot_center = p_door_world
        p_handle_base_world = mech.get_pose_handle_base_world().p
        # TODO: i think the rot_radius needs to be in the rot_center frame (rot_axis)
        #       currently it is in the world frame
        rot_radius = np.subtract(p_handle_base_world, rot_center)
        rot_axis = [0., 0., 0., 1.]
        rot_orientation = [0., 0., 0., 1.]
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation, p_delta)

    @staticmethod
    def _random(bb, p_delta=None):
        """ This function generates a random Revolute policy. The ranges are
        based on the data.generator range prismatic joints
        """
        rot_center = _random_p(bb)
        rot_axis = np.array([0.,0.,0.,1.])
        rot_radius = random_radius()
        rot_orientation = np.array([0.,0.,0.,1.])
        return Revolute(rot_center, rot_axis, rot_radius, rot_orientation, p_delta)

class Poke(Policy):
    def __init__(self, p_delta):
        super(Poke,self).__init__('Poke', p_delta)

class Slide(Policy):
    def __init__(self, p_delta):
        super(Slide,self).__init__('Slide', p_delta)

class Path(Policy):
    def __init__(self, p_delta):
        super(Path,self).__init__('Path', p_delta)

# TODO: add other policy_types as they're made
def generate_random_policy(bb, p_delta=None, policy_types=[Revolute, Prismatic]):
    policy_type = np.random.choice(policy_types)
    return policy_type._random(bb, p_delta)

def generate_model_based_policy(bb, mech, p_delta=None):
    if mech.mechanism_type == 'Door':
        return Revolute._model(bb, mech, p_delta)
    if mech.mechanism_type == 'Slider':
        return Prismatic._model(bb, mech, p_delta)

## Helper Functions
def _random_p(bb):
    bb_center = p.getLinkState(bb.bb_id,0)[0]
    x_limits = np.add(bb_center[0], [-bb.width/2,bb.width/2])
    z_limits = np.add(bb_center[2], [-bb.height/2,bb.height/2])
    x = np.random.uniform(*x_limits)
    # force all positions to lie on busybox backboard
    y = bb.project_onto_backboard([0., 0., 0.,])[1]
    z = np.random.uniform(*z_limits)
    return (x, y, z)

def random_radius():
    # based on generator random radius limits
    r = np.random.uniform(0.05,0.15)
    return np.array([r, 0., 0.])
