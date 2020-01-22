from collections import namedtuple
import numpy as np
from utils import util
import itertools
import pybullet as p

"""See actions.gripper for variable naming and naming conventions
"""

PolicyParams = namedtuple('PolicyParams', 'type params param_data')
"""
Tuple for storing policy data
:param type: str, name of the Policy object type
:param params: one of {actions.policies.PrismaticParams,
                actions.policies.RevoluteParams}
:param param_data: dictionary where keys are param names and values are
                    actions.policies.ParamData
"""

PrismaticParams = namedtuple('PrismaticParams', 'rigid_position \
                                                rigid_orientation \
                                                pitch \
                                                yaw \
                                                goal_config')

RevoluteParams = namedtuple('RevoluteParams', 'rot_center \
                                                rot_axis_roll \
                                                rot_axis_pitch \
                                                rot_axis_yaw \
                                                rot_radius_x \
                                                goal_config')

ParamData = namedtuple('ParamData', 'varied bounds type')
"""
:param varied: boolean, True if param was randomly sampled when policy was generated
:param bounds: list of len()==2, range of values sampled from (if varied == True),
                else range of potential param values
:param type: string 'angular' or 'linear', coordinate system of param for plotting
"""

class Policy(object):
    def __init__(self, type=None):
        """ This is an interface for each Policy type. Each Policy must implement the
        generate_trajectory, _forward_kinematics, _inverse_kinematics, generate_random_config,
        _random, and _model methods
        :param type: string, name of the child Policy class
        """
        self.type = type
        self.traj_lines = []

    def generate_trajectory(self, pose_handle_base_world, debug=False,
                                p_delta= 0.01, color=[0,0,0], old_lines=None):
        """ This method generates a trajectory of waypoints that the gripper tip should
        move through
        :param pose_handle_base_world: util.Pose, initial pose of the base of the handle
        :param debug: if True, display debug visualizations
        :param p_delta: scalar, the distance between trajectory waypoints
        """
        # TODO: don't assume handle always starts at config = 0
        config_curr = self._inverse_kinematics(*pose_handle_base_world)
        config_dir_unit = self._config_dir(config_curr)
        config_delta = p_delta*config_dir_unit

        poses = []
        for i in itertools.count():
            if i < 400:
                if self._past_goal_config(config_curr, config_dir_unit):
                    pose_handle_base_world = self._forward_kinematics(self.goal_config)
                    poses += [pose_handle_base_world]
                    break
                pose_handle_base_world = self._forward_kinematics(config_curr)
                poses += [pose_handle_base_world]
                config_curr += config_delta
            else:
                break
        if debug:
            # draws the planned handle base trajectory
            self._draw_traj(poses, color)
            p.stepSimulation()
        return poses

    def get_policy_tuple(self):
        raise NotImplementedError('get_policy_tuple not implemented for policy \
                                    type '+self.type)

    def _config_dir(self, config_curr):
        return 1 if (self.goal_config > config_curr) else -1

    def _past_goal_config(self, curr_q, q_dir_unit):
        return (curr_q > self.goal_config) if (q_dir_unit == 1) \
                else (curr_q < self.goal_config)

    def _forward_kinematics(self, config):
        raise NotImplementedError('_forward_kinematics function not implemented \
                                    for policy type '+self.type)

    def _inverse_kinematics(self, p_joint_world, q_joint_world):
        raise NotImplementedError('_inverse_kinematics function not implemented \
                                    for policy type '+self.type)

    @staticmethod
    def _get_param_data(policy_type):
        if policy_type == 'Revolute':
            # NOTE: currently the rot_axis_pitch bounds are not used for sampling
            # samples are either 0 or pi just like true doors
            return {'rot_axis_roll': ParamData(False, [0.0, 0.0], 'angular'),
                    'rot_axis_pitch': ParamData(True, [0.0, 2*np.pi], 'angular'),
                    'rot_axis_yaw': ParamData(False, [0.0, 0.0], 'angular'),
                    'radius_x': ParamData(True, [.08-0.025, 0.15], 'linear'),
                    'config': ParamData(True, [-np.pi/2, 0.0], 'linear')}
        elif policy_type == 'Prismatic':
            return {'pitch': ParamData(True, [-np.pi, 0.0], 'angular'),
                    'yaw': ParamData(False, [-np.pi/2, np.pi/2], 'angular'),
                    'config': ParamData(True, [-0.25, 0.25], 'linear')}

    @staticmethod
    def _gen(bb, mech):
        raise NotImplementedError('_gen not implemented for policy type ')

    def _draw_traj(self, poses, color):
        if len(self.traj_lines) > 0:
            for line in self.traj_lines:
                p.removeUserDebugItem(line)
        self._draw(poses, color)

class Prismatic(Policy):
    def __init__(self, pos, orn, pitch, yaw, goal_config, param_data):
        """
        :param pos: vector of length 3, a rigid (x,y,z) position in the world frame
                    along the prismatic joint
        :param orn: vector of length 4, a rigid (x,y,z,w) quaternion in the world
                    frame representing the orientation of the handle in the world
        :param pitch: scalar, pitch between world frame and the direction of the prismatic joint
        :param yaw: scalar, yaw between world frame and the direction of the prismatic joint
        :param goal_config: scalar, distance to move along constrained trajectory
        :param param_delta: dict, keys are param names and values are ParamData tuples
        """
        self.rigid_position = pos
        self.rigid_orientation = orn
        self.pitch = pitch
        self.yaw = yaw
        self.goal_config = goal_config
        self.param_data = param_data

        # derived
        self._M_origin_world = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)
        self.a = util.Pose(self.rigid_position, self.rigid_orientation)
        q_prismatic_dir = util.quaternion_from_euler(0.0, self.pitch, self.yaw)
        self.e = util.transformation([1., 0., 0.], [0., 0., 0.], q_prismatic_dir)
        super(Prismatic, self).__init__('Prismatic')

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

    def get_policy_tuple(self):
        prism_params = PrismaticParams(self.rigid_position,
                                        self.rigid_orientation,
                                        self.pitch,
                                        self.yaw,
                                        self.goal_config)
        return PolicyParams(self.type, prism_params, self.param_data)

    def _draw(self, traj, color):
        lines = util.draw_line([np.add(traj[0].p, [0., .025, 0.]), np.add(traj[-1].p, [0., .025, 0.])], color, thick=False)
        self.traj_lines = lines

    @staticmethod
    def _gen(bb, mech, init_pose=None):
        """ This function generates a Prismatic policy. The ranges are
        based on the data.generator range prismatic joints
        """
        param_data = Policy._get_param_data('Prismatic')

        if not init_pose:
            rigid_position = mech.get_pose_handle_base_world().p
        else:
            rigid_position = init_pose.p
        rigid_orientation = np.array([0., 0., 0., 1.])

        # pitch
        if param_data['pitch'].varied:
            pitch = np.random.uniform(*param_data['pitch'].bounds)
        else:
            if mech.mechanism_type == 'Slider':
                pitch = -np.arctan2(mech.axis[1], mech.axis[0])
            else:
                raise Exception('Cannot use ground truth pitch for Prismatic \
                                policies on non-Prismatic mechanisms as there is \
                                not a single ground truth value. It varies with \
                                each Prismatic mechanism. Must vary pitch \
                                param for non-Prismatic mechanism.')

        # yaw
        if param_data['yaw'].varied:
            yaw = np.random.uniform(*param_data['yaw'].bounds)
        else:
            yaw = 0.0

        # goal config
        if param_data['config'].varied:
            goal_config = np.random.uniform(*param_data['config'].bounds)
        else:
            if mech.mechanism_type == 'Slider':
                goal_config = mech.range/2.0
            else:
                raise Exception('Cannot use ground truth config for Prismatic \
                                policies on non-Prismatic mechanisms as there is \
                                not a single ground truth value. It varies with \
                                each Prismatic mechanism. Must vary config \
                                param for non-Prismatic mechanism.')

        return Prismatic(rigid_position,
                            rigid_orientation,
                            pitch,
                            yaw,
                            goal_config,
                            param_data)

class Revolute(Policy):
    def __init__(self, rot_center, rot_axis_roll, rot_axis_pitch, rot_axis_yaw, \
                    rot_radius_x, goal_config, param_data):
        """
        :param rot_center: vector of length 3, a rigid (x,y,z) position in the world frame
                    of the center of rotation
        :param rot_axis_roll: scalar, roll angle between the world frame to the rotation frame
        :param rot_axis_pitch: scalar, pitch angle between the world frame to the rotation frame
        :param rot_axis_yaw: scalar, yaw angle between the world frame and the rotation frame
        :param rot_radius_x: scalar, distance from handle frame to rotational axis along the -x-axis
                        of the rotation frame
        :param goal_config: scalar, distance to move along constrained trajectory
        :param param_delta: dict, keys are param names and values are ParamData tuples
        """
        self.rot_center = rot_center
        self.rot_axis_roll = rot_axis_roll
        self.rot_axis_pitch = rot_axis_pitch
        self.rot_axis_yaw = rot_axis_yaw
        self.rot_radius_x = rot_radius_x
        self.goal_config = goal_config
        self.param_data = param_data

        # derived
        rot_axis = util.quaternion_from_euler(self.rot_axis_roll, self.rot_axis_pitch, self.rot_axis_yaw)
        rot_orn = [0., 0., 0., 1.] # rotation between handle frame and rotational axis
        self._M_center_world = util.pose_to_matrix(self.rot_center, rot_axis)
        self._M_radius_center = util.pose_to_matrix([self.rot_radius_x, 0., 0.], rot_orn)
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

    def get_policy_tuple(self):
        rev_params = RevoluteParams(self.rot_center,
                                    self.rot_axis_roll,
                                    self.rot_axis_pitch,
                                    self.rot_axis_yaw,
                                    self.rot_radius_x,
                                    self.goal_config)

        return PolicyParams(self.type, rev_params, self.param_data)

    def _draw(self, traj, color):
        lines = []
        for i in range(len(traj)-1):
            lines += util.draw_line([traj[i].p, traj[i+1].p], color, thick=False)
        self.traj_lines = lines

    @staticmethod
    def _gen(bb, mech):
        """ This function generates a Revolute policy. The ranges are
        based on the data.generator range revolute joints
        """
        param_data = Policy._get_param_data('Revolute')

        # axis roll
        if param_data['rot_axis_roll'].varied:
            rot_axis_roll_world = \
                    np.random.uniform(*param_data['rot_axis_roll'].bounds)
        else:
            rot_axis_roll_world = 0.0

        # axis pitch
        if param_data['rot_axis_pitch'].varied:
            # NOTE: this is sampling from a set, not a continuous range
            rot_axis_pitch_world = np.random.choice([0.0, np.pi])
            # rot_axis_pitch_world =
            #        np.random.uniform(*param_data['rot_axis_pitch'].bounds)
        else:
            if mech.mechanism_type == 'Door':
                if not mech.flipped:
                    rot_axis_pitch_world = np.pi
                else:
                    rot_axis_pitch_world = 0.0
            else:
                raise Exception('Cannot use ground truth rot_axis_pitch for Revolute \
                                policies on non-Revolute mechanisms as there is \
                                not a single ground truth value. It varies with \
                                each Revolute mechanism. Must vary rot_axis_pitch \
                                param for non-Revolute mechanism.')
        # axis yaw
        if param_data['rot_axis_yaw'].varied:
            rot_axis_yaw_world = \
                    np.random.uniform(*param_data['rot_axis_yaw'].bounds)
        else:
            rot_axis_yaw_world = 0.0

        # radius_x
        if param_data['radius_x'].varied:
            radius_x = np.random.uniform(*param_data['radius_x'].bounds)
        else:
            if mech.mechanism_type == 'Door':
                radius_x = mech.get_radius_x()
            else:
                raise Exception('Cannot use ground truth radius_x for Revolute \
                                policies on non-Revolute mechanisms as there is \
                                not a single ground truth value. It varies with \
                                each Revolute mechanism. Must vary radius_x \
                                param for non-Revolute mechanism.')

        # center of rotation
        rot_axis_world = util.quaternion_from_euler(rot_axis_roll_world, rot_axis_pitch_world, rot_axis_yaw_world)
        radius = [-radius_x, 0.0, 0.0]
        p_handle_base_world = mech.get_pose_handle_base_world().p
        p_rot_center_world = p_handle_base_world + util.transformation(radius, [0., 0., 0.], rot_axis_world)

        # goal config
        if param_data['config'].varied:
            goal_config = np.random.uniform(*param_data['config'].bounds)
        else:
            if mech.mechanism_type == 'Door':
                goal_config = -np.pi/2
            else:
                raise Exception('Cannot use ground truth config for Revolute \
                                policies on non-Revolute mechanisms as there is \
                                not a single ground truth value. It varies with \
                                each Revolute mechanism. Must vary config \
                                param for non-Revolute mechanism.')

        return Revolute(p_rot_center_world,
                        rot_axis_roll_world,
                        rot_axis_pitch_world,
                        rot_axis_yaw_world,
                        radius_x,
                        goal_config,
                        param_data)

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
def generate_policy(bb, mech, random_policies, policy_types=[Revolute, Prismatic], init_pose=None):
    if not random_policies:
        policy_type = get_matched_policy_type(mech)
        if policy_type == 'Revolute':
            return Revolute._gen(bb, mech)
        elif policy_type == 'Prismatic':
            return Prismatic._gen(bb, mech, init_pose=init_pose)
    else:
        policy_type = np.random.choice(policy_types)
        return policy_type._gen(bb, mech)

def get_matched_policy_type(mech):
    if mech.mechanism_type == 'Door':
        return 'Revolute'
    elif mech.mechanism_type == 'Slider':
        return 'Prismatic'

def get_policy_from_tuple(policy_params):
    type = policy_params.type
    params = policy_params.params
    param_data = policy_params.param_data
    if policy_params.type == 'Revolute':
        policy = Revolute(params.rot_center, params.rot_axis_roll, params.rot_axis_pitch,
                        params.rot_axis_yaw, params.rot_radius_x, params.goal_config)
    if policy_params.type == 'Prismatic':
        policy = Prismatic(params.rigid_position, params.rigid_orientation, params.pitch,
                        params.yaw, params.goal_config)
    policy.param_data = param_data
    return policy

## Helper Functions
def _random_p(bb):
    bb_center = bb.get_center_pos()
    x_limits = np.add(bb_center[0], [-bb.width/2,bb.width/2])
    z_limits = np.add(bb_center[2], [-bb.height/2,bb.height/2])
    x = np.random.uniform(*x_limits)
    # force all positions to lie on busybox backboard
    y = bb.project_onto_backboard([0., 0., 0.,])[1]
    z = np.random.uniform(*z_limits)
    return (x, y, z)
