from collections import namedtuple
import numpy as np
from util import util
import itertools
import pybullet as p

"""See actions.gripper for variable naming and naming conventions
"""

PolicyParams = namedtuple('PolicyParams', 'type params delta_values')
"""
Tuple for storing policy data
:param type: str, name of the Policy object type
:param params: one of {actions.policies.PrismaticParams, actions.policies.RevoluteParams}
:param delta_values: one of {actions.policies.PrismaticDelta, actions.policies.RevoluteDelta}
"""

PrismaticParams = namedtuple('PrismaticParams', 'rigid_position rigid_orientation pitch yaw')
PrismaticDelta = namedtuple('PrismaticDelta', 'delta_pitch delta_yaw')

RevoluteParams = namedtuple('RevoluteParams', 'rot_center rot_roll rot_pitch rot_radius rot_orientation')
RevoluteDelta = namedtuple('RevoluteDelta', 'delta_roll, delta_pitch, delta_radius_x, delta_radius_z')

class Policy(object):
    def __init__(self, type=None):
        """ This is an interface for each Policy type. Each Policy must implement the
        generate_trajectory, _forward_kinematics, _inverse_kinematics, generate_random_config,
        _random, and _model methods
        :param type: string, name of the child Policy class
        """
        self.type = type

    def generate_trajectory(self, pose_handle_base_world, config_goal, debug, p_delta= 0.01):
        """ This method generates a trajectory of waypoints that the gripper tip should
        move through
        :param pose_handle_base_world: util.Pose, initial pose of the base of the handle
        :param config_goal: the goal configuration of the joint
        :param debug: if True, display debug visualizations
        :param p_delta: scalar, the distance between trajectory waypoints
        """
        # TODO: don't assume handle always starts at config = 0
        config_curr = self._inverse_kinematics(*pose_handle_base_world)
        config_dir_unit = self._config_dir(config_curr, config_goal)
        config_delta = p_delta*config_dir_unit

        poses = []
        for i in itertools.count():
            if self._past_goal_config(config_curr, config_goal, config_dir_unit):
                pose_handle_base_world = self._forward_kinematics(config_goal)
                poses += [pose_handle_base_world]
                break
            pose_handle_base_world = self._forward_kinematics(config_curr)
            poses += [pose_handle_base_world]
            config_curr += config_delta
        if debug:
            # draws the planned handle base trajectory
            self._draw_traj(poses)
        return poses

    @staticmethod
        def generate_config(mech, goal_config):
        raise NotImplementedError('generate_config not implemented for policy type '+self.type)

    def get_policy_tuple(self):
        raise NotImplementedError('get_policy_tuple not implemented for policy type '+self.type)

    def _config_dir(self, config_curr, config_goal):
        return 1 if (config_goal > config_curr) else -1

    def _past_goal_config(self, curr_q, config_goal, q_dir_unit):
        return (curr_q > config_goal) if (q_dir_unit == 1) else (curr_q < config_goal)

    def _forward_kinematics(self, config):
        raise NotImplementedError('_forward_kinematics function not implemented for policy type '+self.type)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        raise NotImplementedError('_inverse_kinematics function not implemented for policy type '+self.type)

    @staticmethod
    def _gen(bb, mech, randomness):
        raise NotImplementedError('_gen not implemented for policy type '+self.type)

class Prismatic(Policy):
    def __init__(self, pos, orn, pitch, yaw, delta_pitch=None, delta_yaw=None):
        """
        :param pos: vector of length 3, a rigid (x,y,z) position in the world frame
                    along the prismatic joint
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion in the world
                    frame representing the orientation of the handle in the world
        :param pitch: scalar, pitch between world frame and the direction of the prismatic joint
        :param yaw: scalar, yaw between world frame and the direction of the prismatic joint
        :param delta_pitch: scalar or None, distance from model-based true pitch for Slider, else None
        :param delta_yaw: scalar or None, distance from model-based true yaw for Slider, else None
        """
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.pitch = pitch
        self.yaw = yaw
        self.delta_pitch = delta_pitch
        self.delta_yaw = delta_yaw

        # derived
        self._M_origin_world = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)
        super(Prismatic,self).__init__('Prismatic')

    def _forward_kinematics(self, config):
        q_prismatic_dir = util.quaternion_from_euler(0.0, self.pitch, self.yaw)
        prismatic_dir = util.transformation([1., 0., 0.], [0., 0., 0.], q_prismatic_dir)
        p_joint_origin = np.multiply(config, prismatic_dir)
        p_joint_origin_4 = np.concatenate([p_joint_origin, [1.]])
        p_joint_world = np.dot(self._M_origin_world, p_joint_origin_4)[:3]
        q_joint_world = util.quaternion_from_matrix(self._M_origin_world)
        return util.Pose(p_joint_world, q_joint_world)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        q_prismatic_dir = util.quaternion_from_euler(0.0, self.pitch, self.yaw)
        prismatic_dir = util.transformation([1., 0., 0.], [0., 0., 0.], q_prismatic_dir)
        M_joint_world = util.pose_to_matrix(p_joint_world, q_joint_world)
        M_world_origin = np.linalg.inv(self._M_origin_world)
        M_joint_origin = np.dot(M_world_origin, M_joint_world)
        p_joint_origin = M_joint_origin[:3,3]
        return np.dot(prismatic_dir, p_joint_origin)

    @staticmethod
    def generate_config(mech, goal_config):
        if goal_config is None:
            return np.random.uniform(-0.25,0.25) # from gen.generator_busybox range limits
        else:
            return goal_config*mech.range/2.0

    def get_policy_tuple(self):
        prism_params = PrismaticParams(self.rigid_position, self.rigid_orientation, self.pitch, self.yaw)
        delta_values = PrismaticDelta(self.delta_pitch, self.delta_yaw)
        return PolicyParams(self.type, prism_params, delta_values)

    def _draw_traj(self, traj):
        for i in range(len(traj)-1):
            # raise so can see above track
            p.addUserDebugLine(np.add(traj[i].p, [0., .025, 0.]), np.add(traj[i+1].p, [0., .025, 0.]))

    @staticmethod
    def _gen(bb, mech, randomness):
        """ This function generates a Prismatic policy. The ranges are
        based on the data.generator range prismatic joints
        """
        p_handle_world = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        rigid_position = bb.project_onto_backboard(p_handle_world)
        rigid_orientation = np.array([0., 0., 0., 1.])
        if mech.mechanism_type == 'Slider':
            pitch = -np.arccos(mech.axis[0])
            yaw = 0.0
        else:
            raise NotImplementedError('Still need to implement random Prismatic for Door')
        delta_pitch = randomness*np.random.uniform(-np.pi/2, np.pi/2)
        delta_yaw = randomness*np.random.uniform(-np.pi/2, np.pi/2)
        return Prismatic(rigid_position, rigid_orientation, pitch+delta_pitch,
                yaw+delta_yaw, delta_pitch, delta_yaw)

class Revolute(Policy):
    def __init__(self, center, axis_roll, axis_pitch, radius, orn, delta_roll=None,
                    delta_pitch=None, delta_radius_x=None, delta_radius_z=None):
        """
        :param center: vector of length 3, a rigid (x,y,z) position in the world frame
                    of the center of rotation
        :param axis_roll: scalar, roll angle between the world frame to the rotation frame
        :param axis_pitch: scalar, pitch angle between the world frame to the rotation frame
        :param radius: vector of length 3, an (x,y,z) position of the radius/handle base
                        in the rotation frame
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion representing the
                    rotataion from rotation frame to the handle base frame
        :param delta_roll: scalar or None, distance from true rotation frame roll for Door, else None
        :param delta_pitch: scalar or None, distance from true rotation frame pitch for Door, else None
        :param delta_radius_x: scalar or None, distance from true revolute radius in x-direction for Door, else None
        :param delta_radius_z: scalar or None, distance from true revolute radius in z-direction for Door, else None
        """
        self.rot_center = center
        self.rot_axis_roll = axis_roll
        self.rot_axis_pitch = axis_pitch
        self.rot_radius = radius
        self.rot_orientation = orn
        self.delta_roll = delta_roll
        self.delta_pitch = delta_pitch
        self.delta_radius_x = delta_radius_x
        self.delta_radius_z = delta_radius_z

        # derived
        rot_axis = util.quaternion_from_euler(self.rot_axis_roll, self.rot_axis_pitch, 0.0)
        self._M_center_world = util.pose_to_matrix(self.rot_center, rot_axis)
        self._M_radius_center = util.pose_to_matrix(self.rot_radius, self.rot_orientation)
        super(Revolute,self).__init__('Revolute')

    def _forward_kinematics(self, config):
        # rotation matrix for a rotation about the z-axis by config radians
        M_joint_z = util.trans.rotation_matrix(-config,[0,0,1])
        M_joint_world = util.trans.concatenate_matrices(self._M_center_world,M_joint_z,self._M_radius_center)
        p_joint_world = M_joint_world[:3,3]
        q_joint_world = util.quaternion_from_matrix(M_joint_world)
        return util.Pose(p_joint_world, q_joint_world)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        # this is only used at the beginning of generate trajectory to get the initial
        # configuration. for now hard code so starts at config=0 (when handle frame is
        # aligned with rot center frame)
        rot_axis = util.quaternion_from_euler(self.rot_axis_roll, self.rot_axis_pitch, 0.0)
        M_joint_world = util.pose_to_matrix(p_joint_world, rot_axis)
        M_joint_center = np.dot(np.linalg.inv(M_joint_world),self._M_center_world)
        # transformation from the radius in the center to the joint in the center
        M_radius_joint_center = np.dot(np.linalg.inv(self._M_radius_center),M_joint_center)
        angle, direction, point = util.trans.rotation_from_matrix(M_radius_joint_center)
        return angle

    @staticmethod
    def generate_config(mech, goal_config):
        if goal_config is None:
            return np.random.uniform(-np.pi/2,np.pi/2)
        else:
            return goal_config*np.pi/2.0

    def get_policy_tuple(self):
        rev_params = RevoluteParams(self.rot_center, self.rot_axis_roll, \
                        self.rot_axis_pitch, self.rot_radius, self.rot_orientation)
        delta_values = RevoluteDelta(self.delta_roll, self.delta_pitch, self.delta_radius_x, \
                        self.delta_radius_z)
        return PolicyParams(self.type, rev_params, delta_values)

    def _draw_traj(self, traj):
        for i in range(len(traj)-1):
            p.addUserDebugLine(traj[i].p, traj[i+1].p)

    @staticmethod
    def _gen(bb, mech, randomness):
        """ This function generates a Revolute policy. The ranges are
        based on the data.generator range revolute joints
        """
        rot_axis_roll = 0.0
        rot_axis_pitch = 0.0
        delta_roll = randomness*np.random.uniform(-np.pi/2, np.pi/2)
        delta_pitch = randomness*np.random.uniform(-np.pi/2, np.pi/2)
        rot_axis = util.quaternion_from_euler(rot_axis_roll+delta_roll,rot_axis_pitch+delta_pitch, 0.0)
        p_handle_base_world = mech.get_pose_handle_base_world().p
        if mech.mechanism_type == 'Door':
            rot_center_true = p.getLinkState(bb.bb_id, mech.door_base_id)[0]
        elif mech.mechanism_type == 'Slider':
            raise NotImplementedError('need to implement random revolute policy for slider')
        rot_radius_world_true = np.subtract(p_handle_base_world, rot_center_true)
        rot_radius_world = rot_radius_world_true
        delta_radius_x = randomness*np.random.uniform(-.8, .8)
        rot_radius_world[0] = rot_radius_world_true[0] + delta_radius_x
        # TODO: see if making this random actually has an effect
        delta_radius_z = randomness*np.random.uniform(-.4, .4)
        rot_radius_world[2] = rot_radius_world_true[2] + delta_radius_z
        rot_radius = util.transformation(rot_radius_world, [0., 0., 0.,], rot_axis, inverse=True)
        rot_center = np.subtract(p_handle_base_world, rot_radius_world)
        # assume no rotation between center frame and handle frame
        rot_orientation = [0.,0.,0.,1.]
        return Revolute(rot_center, rot_axis_roll, rot_axis_pitch, rot_radius, rot_orientation, \
                delta_roll, delta_pitch, delta_radius_x, delta_radius_z)

class Poke(Policy):
    def __init__(self):
        super(Poke,self).__init__('Poke')

class Slide(Policy):
    def __init__(self):
        super(Slide,self).__init__('Slide')

class Path(Policy):
    def __init__(self):
        super(Path,self).__init__('Path')

# TODO: add other policy_types as they're made
def generate_policy(bb, mech, match_policies, randomness, policy_types=[Revolute, Prismatic]):
    if match_policies:
        if mech.mechanism_type == 'Door':
            return Revolute._gen(bb, mech, randomness)
        elif mech.mechanism_type == 'Slider':
            return Prismatic._gen(bb, mech, randomness)
    else:
        policy_type = np.random.choice(policy_types)
        return policy_type._gen(bb, mech, randomness)

def get_policy_from_params(type, params):
    if type == 'Revolute':
        return Revolute(params[:3], params[3:7], params[7:10], params[10:14])
    if type == 'Prismatic':
        return Prismatic(params[:3], params[3:7], params[7:10])

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
