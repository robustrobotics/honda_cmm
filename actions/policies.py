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
:param params: one of {actions.policies.PrismaticParams, actions.policies.RevoluteParams}
:param vary_params: dictionary with keys corresponding to Policy subclasses (string)
                    and keys corresponding to dictionaries with policy parameter
                    keys and True/False values where True means the param was
                    randomly sampled when generating this policy
"""

PrismaticParams = namedtuple('PrismaticParams', 'rigid_position rigid_orientation pitch yaw')

RevoluteParams = namedtuple('RevoluteParams', 'rot_center rot_axis_roll rot_axis_pitch \
                            rot_axis rot_radius_x rot_orientation')
ParamData = namedtuple('ParamData', 'name varied bounds type')
"""
:param name: string, name of param
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
        self.param_data = {'Revolute': [ParamData('rot_axis_roll', False, [0.0, 0.0], 'angular'),
                                        ParamData('rot_axis_pitch', True, [0.0, 2*np.pi], 'angular'),
                                        ParamData('rot_axis_yaw', False, [0.0, 0.0], 'angular'),
                                        ParamData('radius_x', True, [.08-0.025, 0.15], 'linear'),
                                        ParamData('config', True, [-np.pi/2, 0.0], 'linear')],
                            'Prismatic': [ParamData('pitch', True, [-np.pi, 0.0], 'angular'),
                                        ParamData('yaw', False, [-np.pi/2, np.pi/2], 'angular'),
                                        ParamData('config', True, [-0.25, 0.25], 'linear')]}

    def generate_trajectory(self, pose_handle_base_world, config_goal, debug=False, p_delta= 0.01, color=[0,0,0], old_lines=None):
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
            if i < 400:
                if self._past_goal_config(config_curr, config_goal, config_dir_unit):
                    pose_handle_base_world = self._forward_kinematics(config_goal)
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

    @staticmethod
    def generate_config(mech, goal_config):
        raise NotImplementedError('generate_config not implemented for policy type ')

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
        raise NotImplementedError('_gen not implemented for policy type ')

    @staticmethod
    def get_plot_data(policy_type):
        """
        Return a list of PolicyPlotData for each policy type.
        This depends on the parameters that are passed into the Policy Encoder.
        """
        # these values are the mins and maxes generated in the _gen() functions
        if policy_type == 'Prismatic':
            return [PolicyPlotData('pitch', 0, [-np.pi, 0], 'angular'),
                    PolicyPlotData('yaw', 1, [-np.pi/2, np.pi/2], 'angular'),
                    PolicyPlotData('config', 2, [-0.25, 0.25], 'linear')]
        elif policy_type == 'Revolute':
            return [PolicyPlotData('roll', 0, [0, 2*np.pi], 'angular'),
                    PolicyPlotData('pitch', 1, [0, 0], 'angular'),
                    PolicyPlotData('radius', 2, [.08-0.025, 0.15], 'linear'),
                    PolicyPlotData('config', 3, [-np.pi/2, 0.0], 'linear')]

    def _draw_traj(self, poses, color):
        if len(self.traj_lines) > 0:
            for line in self.traj_lines:
                p.removeUserDebugItem(line)
        self._draw(poses, color)

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

        # should calc delta pitch from mech params and pitch, yaw
        self.delta_pitch = delta_pitch
        self.delta_yaw = delta_yaw

        # mask to specify which parameters should be varied
        self.mask = {'pitch': True, 'yaw': False, 'q': True}

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

    @staticmethod
    def generate_config(mech, goal_config):
        if goal_config is None:
            max_config = 0.25 # from generator_busybox
            # max_config = np.random.uniform(-0.1, 0.1)
            return np.random.uniform(-max_config, max_config)
        else:
            return goal_config*mech.range/2.0

    def get_policy_tuple(self):
        prism_params = PrismaticParams(self.rigid_position, self.rigid_orientation, self.pitch, self.yaw)
        delta_values = PrismaticDelta(self.delta_pitch, self.delta_yaw)
        return PolicyParams(self.type, prism_params, delta_values)

    def _draw(self, traj, color):
        lines = util.draw_line([np.add(traj[0].p, [0., .025, 0.]), np.add(traj[-1].p, [0., .025, 0.])], color, thick=False)
        self.traj_lines = lines

    @staticmethod
    def _gen(bb, mech, randomness, init_pose=None):
        """ This function generates a Prismatic policy. The ranges are
        based on the data.generator range prismatic joints
        """
        if not init_pose:
            rigid_position = mech.get_pose_handle_base_world().p
        else:
            rigid_position = init_pose.p
        rigid_orientation = np.array([0., 0., 0., 1.])
        if mech.mechanism_type == 'Slider':
            true_pitch = -np.arctan2(mech.axis[1], mech.axis[0])
            true_yaw = 0.0

            delta_pitch = randomness*np.random.uniform(-np.pi/2, np.pi/2)
            delta_yaw = randomness*np.random.uniform(-np.pi/2, np.pi/2)

            pitch = true_pitch + delta_pitch
            yaw = true_yaw + delta_yaw

            if pitch < -np.pi:
                pitch += np.pi
            elif pitch > 0:
                pitch -= np.pi
            # TODO: same for yaw if have mech with yaw != 0
        else:
            pitch = np.random.uniform(-np.pi, 0.0)
            yaw = np.random.uniform(-np.pi/2, np.pi/2)
            delta_pitch = None
            delta_yaw = None
        return Prismatic(rigid_position, rigid_orientation, pitch, yaw, \
                delta_pitch, delta_yaw)


    def get_goal_from_policy(self, goal_config):#, bb, mech, handle_pose, goal_pos):
        goal_pose = self._forward_kinematics(goal_config)
        return goal_pose.p

    @staticmethod
    def get_policy_from_goal(bb, mech, handle_pose, goal_pos):
        direction3d = np.subtract(goal_pos, handle_pose.p)
        direction = [direction3d[0], direction3d[2]]
        dist = np.linalg.norm(direction)
        axis = np.divide(direction, dist)
        # mechs all have positive z axis
        if axis[1] < 0:
            axis = -1*axis
        p.addUserDebugLine(handle_pose.p, np.add(handle_pose.p, [axis[0], 0.0, axis[1]]), [0,1,0], lifeTime=0)
        pitch = -np.arctan2(axis[1], axis[0])
        yaw = 0.0
        true_policy = Prismatic._gen(bb, mech, 0.0)
        delta_pitch = pitch - true_policy.pitch
        delta_yaw = yaw - true_policy.yaw
        if delta_pitch > np.pi/2:
            delta_pitch = np.pi - delta_pitch
        elif delta_pitch < -np.pi/2:
            delta_pitch = -np.pi - delta_pitch
        # TODO: same for yaw if have mech with yaw != 0
        rigid_position = handle_pose.p
        rigid_orientation = np.array([0., 0., 0., 1.])
        policy = Prismatic(rigid_position, rigid_orientation, pitch, yaw, delta_pitch,
                delta_yaw)
        goal_config = policy._inverse_kinematics(goal_pos, [0., 0., 0., 1.])
        return policy, goal_config

class Revolute(Policy):
    def __init__(self, rot_center, rot_axis_roll, rot_axis_pitch, rot_axis_yaw, rot_radius_x):
        """
        :param center: vector of length 3, a rigid (x,y,z) position in the world frame
                    of the center of rotation
        :param rot_axis_roll: scalar, roll angle between the world frame to the rotation frame
        :param rot_axis_pitch: scalar, pitch angle between the world frame to the rotation frame
        :param rot_axis_yaw: scalar, yaw angle between the world frame and the rotation frame
        :param rot_radius_x: scalar, distance from handle frame to rotational axis along the -x-axis
                        of the rotation frame
        """
        self.rot_center = rot_center
        self.rot_axis_roll = rot_axis_roll
        self.rot_axis_pitch = rot_axis_pitch
        self.rot_axis_yaw = rot_axis_yaw
        self.rot_radius_x = rot_radius_x

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

    @staticmethod
    def generate_config(mech, goal_config):
        if goal_config is None:
            return -np.random.uniform(0.0, np.pi/2)
        else:
            return -goal_config*np.pi/2.0

    def get_policy_tuple(self):
        rev_params = RevoluteParams(self.rot_center, self.rot_axis_roll, \
                        self.rot_axis_pitch, self.rot_axis, self.rot_radius_x, \
                        self.rot_orientation)
        delta_values = RevoluteDelta(self.delta_roll, self.delta_pitch, self.delta_radius_x)
        return PolicyParams(self.type, rev_params, delta_values)

    def _draw(self, traj, color):
        lines = []
        for i in range(len(traj)-1):
            lines += util.draw_line([traj[i].p, traj[i+1].p], color, thick=False)
        self.traj_lines = lines

    @staticmethod
    def _gen(bb, mech, randomness):
        """ This function generates a Revolute policy. The ranges are
        based on the data.generator range revolute joints
        """
        if randomness == 0 and mech.mechanism_type != 'Door':
            raise Exception('cannot set randomness == 0 and try Revolute policy \
                                on non-Revolute mechanism')
        elif randomness == 0 and mech.mechanism_type == 'Door':
            Policy.vary_param['Revolute'] = {'axis_roll': False, 'axis_pitch': False,
                                            'axis_yaw': False, 'radius_x': False}

        # set roll param
        if Policy.vary_param['Revolute']['rot_axis_roll']:
            rot_axis_roll_world = np.random.uniform(0.0, 2 * np.pi)
        else:
            rot_axis_roll_world = 0.0
        # set pitch param
        if Policy.vary_param['Revolute']['rot_axis_pitch']:
            rot_axis_pitch_world = np.random.choice([0.0, np.pi])
            # rot_axis_pitch_world = np.random.uniform(0.0, 2*np.pi)
        else:
            if mech.mechanism_type == 'Door':
                if not mech.flipped:
                    rot_axis_pitch_world = np.pi
                else:
                    rot_axis_pitch_world = 0.0
            else:
                raise Exception('cannot mask the rot_axis_pitch when attempting \
                        Revolute policies on non-Revolute mechanisms')
        # set yaw param
        if Policy.vary_param['Revolute']['rot_axis_yaw']:
            rot_axis_yaw_world = np.random.uniform(0.0, 2*np.pi)
        else:
            rot_axis_yaw_world = 0.0
        # set radius_x param
        if Policy.vary_param['Revolute']['radius_x']:
            radius_x = np.random.uniform(0.08-0.025, 0.15)
        else:
            if mech.mechanism_type == 'Door':
                radius_x = mech.get_radius_x()
            else:
                raise Exception('cannot mask the radius_x when attempting \
                        Revolute policies on non-Revolute mechanisms')

        # calculate the center of rotation in the world frame
        rot_axis_world = util.quaternion_from_euler(rot_axis_roll_world, rot_axis_pitch_world, rot_axis_yaw_world)
        radius = [-radius_x, 0.0, 0.0]
        p_handle_base_world = mech.get_pose_handle_base_world().p
        p_rot_center_world = p_handle_base_world + util.transformation(radius, [0., 0., 0.], rot_axis_world)

        return Revolute(p_rot_center_world,
                        rot_axis_roll_world,
                        rot_axis_pitch_world,
                        rot_axis_yaw_world,
                        radius_x)

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
def generate_policy(bb, mech, random_policies, randomness, policy_types=[Revolute, Prismatic], init_pose=None):
    if not random_policies:
        policy_type = get_matched_policy_type(mech)
        if policy_type == 'Revolute':
            return Revolute._gen(bb, mech, randomness)
        elif policy_type == 'Prismatic':
            return Prismatic._gen(bb, mech, randomness, init_pose=init_pose)
    else:
        policy_type = np.random.choice(policy_types)
        return policy_type._gen(bb, mech, randomness)

def get_matched_policy_type(mech):
    if mech.mechanism_type == 'Door':
        return 'Revolute'
    elif mech.mechanism_type == 'Slider':
        return 'Prismatic'

def get_policy_from_tuple(policy_params):
    type = policy_params.type
    params = policy_params.params
    delta_values = policy_params.delta_values
    if policy_params.type == 'Revolute':
        return Revolute(params.rot_center, params.rot_axis_roll, params.rot_axis_pitch,
                        params.rot_axis, params.rot_radius_x, params.rot_orientation,
                        delta_values.delta_roll, delta_values.delta_pitch, delta_values.delta_radius_x)
    if policy_params.type == 'Prismatic':
        return Prismatic(params.rigid_position, params.rigid_orientation, params.pitch,
                        params.yaw, delta_values.delta_pitch, delta_values.delta_yaw)

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
