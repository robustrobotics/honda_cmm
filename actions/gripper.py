import pybullet as p
import numpy as np
from util import util
from collections import namedtuple
import itertools
import sys

"""
Naming convention
-----------------
pose_ is a util.Pose()
M_ is a transformation matrix
p_ is a vector of length 3 representing a position
q_ is a vector of length 4 representing a quaternion
e_ is a vector of length 3 representing euler angles
v_ is a vector of length 6 with [0:3] representing the linear velocity and
                                [3:6] representing the angular velocity
lin_v_ is a vector of length 3 representing the linear velocity
omega_ is a vector of length 3 representing the angular velocity
config_ is the configuration of a joint. the dimensionality depends on the joint-type
the first following variable name represents the point/pose being described
the second following variable name indicates the frame that the point or pose is defined in

Variables
---------
world - world frame
base - gripper base frame
tip - tip of the gripper frame
com - center of mass of the entire gripper body
left - left finger tip of gripper
right - right finger tip of gripper
joint - mechanism handle
joint_base - handle projected onto the BusyBox back link
door - door base of a revolute joint
track - track of prismatic joint
_err - error
_des - desired
_thresh - threshold
_M - matrix form of a pose/transformation
"""

class Gripper:
    def __init__(self, bb_id, k=[2000.0,20.0], d=[0.45,0.45], add_dist=0.1, p_err_thresh=0.005):
        """
        This class defines the actions a gripper can take such as grasping a handle
        and executing PD control
        :param bb_id: int, the pybullet id of the BusyBox
        :param k: a vector of length 2 where the first entry is the linear position
                    (stiffness) gain and the second entry is the angular position gain
        :param d: a vector of length 2 where the first entry is the linear derivative
                    (damping) gain and the second entry is the angular derivative gain
        :param add_dist: scalar, the distance the PD controller is trying to control to
        :param p_err_thresh: scalar, the allowable error before the controller moves
                                to the next setpoint
        """
        self._id = p.loadSDF("../models/gripper/gripper_high_fric.sdf")[0]
        self._bb_id = bb_id
        self._left_finger_tip_id = 2
        self._right_finger_tip_id = 5
        self._left_finger_base_joint_id = 0
        self._right_finger_base_joint_id = 3
        self._finger_force = 20
        self.pose_tip_world_reset = util.Pose([0.0, 0.0, 0.2], \
                            [0.50019904,  0.50019904, -0.49980088, 0.49980088])

        # get mass of gripper
        mass = 0
        for link in range(p.getNumJoints(self._id)):
            mass += p.getDynamicsInfo(self._id, link)[0]
        self._mass = mass

        # control parameters
        self.k = k
        self.d = d
        self.add_dist = add_dist
        self.p_err_thresh = p_err_thresh

    def _get_p_tip_world(self):
        p_left_world = p.getLinkState(self._id, self._left_finger_tip_id)[0]
        p_right_world = p.getLinkState(self._id, self._right_finger_tip_id)[0]
        p_tip_world = np.mean([p_left_world, p_right_world], axis=0)
        return p_tip_world

    def _get_p_tip_base(self):
        p_base_world, q_base_world = p.getBasePositionAndOrientation(self._id)
        p_tip_world = self._get_p_tip_world()
        p_tip_base = util.transformation(p_tip_world, p_base_world, q_base_world, inverse=True)
        return p_tip_base

    def _get_pose_com_(self, frame):
        com_numerator = np.array([0.0, 0.0, 0.0])
        for link_index in range(p.getNumJoints(self._id)):
            link_com = p.getLinkState(self._id, link_index)[0]
            link_mass = p.getDynamicsInfo(self._id, link_index)[0]
            com_numerator = np.add(com_numerator, np.multiply(link_mass,link_com))
        p_com_world = np.divide(com_numerator, self._mass)

        p_base_world, q_base_world = p.getBasePositionAndOrientation(self._id)
        q_com_world = q_base_world

        if frame == 'world':
            return p_com_world, q_com_world
        elif frame == 'tip':
            p_tip_world = self._get_p_tip_world()
            p_com_tip = util.transformation(p_com_world, p_tip_world, q_base_world, inverse=True)
            q_com_tip = np.array([0.,0.,0.,1.])
            return p_com_tip, q_com_tip
        elif frame == 'base':
            p_com_base = util.transformation(p_com_world, p_base_world, q_base_world, inverse=True)
            q_com_base = np.array([0.0,0.0,0.0,1.0])
            return p_com_base, q_com_base

    def _get_v_com_world_error(self, v_tip_world_des):
        p_com_tip, q_com_tip = self._get_pose_com_('tip')
        v_com_world_des = util.adjoint_transformation(v_tip_world_des, p_com_tip, q_com_tip, inverse=True)

        v_base_world = np.concatenate(p.getBaseVelocity(self._id))
        p_com_base, q_com_base = self._get_pose_com_('base')
        v_com_world = util.adjoint_transformation(v_base_world, p_com_base, q_com_base, inverse=True)

        v_com_world_err = np.subtract(v_com_world_des, v_com_world)
        return v_com_world_err[:3], v_com_world_err[3:]

    def _get_pose_handle_base_world_error(self, pose_handle_base_world_des, q_offset, mech):
        pose_handle_base_world = mech.get_pose_handle_base_world()
        p_handle_base_world_err = np.subtract(pose_handle_base_world_des.p, pose_handle_base_world.p)

        q_handle_base_world_des = util.quat_math(pose_handle_base_world_des.q, q_offset, False, False)
        q_handle_base_world_err = util.quat_math(q_handle_base_world_des, pose_handle_base_world.q, False, True)
        e_handle_base_world_err = util.euler_from_quaternion(q_handle_base_world_err)
        return p_handle_base_world_err, e_handle_base_world_err

    def _at_des_handle_base_pose(self, pose_handle_base_world_des, q_offset, mech):
        p_handle_base_world_err, _ = self._get_pose_handle_base_world_error(pose_handle_base_world_des, q_offset, mech)
        return np.linalg.norm(p_handle_base_world_err) < self.p_err_thresh

    def _in_contact(self, mech):
        points = p.getContactPoints(self._id, self._bb_id, linkIndexB=mech.handle_id)
        if len(points)>0:
            return True
        return False

    def _set_pose_tip_world(self, pose_tip_world_des, reset=False):
        p_base_tip = np.multiply(-1, self._get_p_tip_base())
        p_base_world_des = util.transformation(p_base_tip, pose_tip_world_des.p, pose_tip_world_des.q)
        p.resetBasePositionAndOrientation(self._id, p_base_world_des, pose_tip_world_des.q)
        p.stepSimulation()

    def _grasp_handle(self, pose_tip_world_des, debug=False):
        # move to default start pose
        for t in range(10):
            self._set_pose_tip_world(self.pose_tip_world_reset, reset=True)

        # open fingers
        for t in range(10):
            self._control_fingers('open', debug=debug)

        # move to desired pose
        for t in range(10):
            self._set_pose_tip_world(pose_tip_world_des, reset=True)

        # close fingers
        for t in range(10):
            self._control_fingers('close', debug=debug)

    def _control_fingers(self, finger_state, debug=False):
        if finger_state == 'open':
            finger_angle = 0.2
        elif finger_state == 'close':
            finger_angle = 0.0
        p.setJointMotorControl2(self._id,self._left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle,force=self._finger_force)
        p.setJointMotorControl2(self._id,self._right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle,force=self._finger_force)
        p.setJointMotorControl2(self._id,2,p.POSITION_CONTROL,targetPosition=0,force=self._finger_force)
        p.setJointMotorControl2(self._id,5,p.POSITION_CONTROL,targetPosition=0,force=self._finger_force)
        p.stepSimulation()

    def _move_PD(self, pose_handle_base_world_des, prev_pose_handle_base_world_des, q_offset, mech, debug=False, timeout=100):
        # move setpoint further away in a straight line between previous desired pose and current desired pose
        if prev_pose_handle_base_world_des is None:
            # TODO: could do something smarter here. this assumes initially want to move in +x direction
            dir = [0,1,0]
            unit_dir = np.divide(dir,np.linalg.norm([dir]))
            p_handle_base_world_des_far = np.add(pose_handle_base_world_des.p, np.multiply(self.add_dist,unit_dir))
        else:
            dir = np.subtract(pose_handle_base_world_des.p, prev_pose_handle_base_world_des.p)
            unit_dir = np.divide(dir,np.linalg.norm([dir]))
            p_handle_base_world_des_far = np.add(prev_pose_handle_base_world_des.p, np.multiply(self.add_dist,unit_dir))
        pose_handle_base_world_des_far = util.Pose(p_handle_base_world_des_far, pose_handle_base_world_des.q)

        finished = False
        handle_base_ps = []

        if debug:
            p.addUserDebugLine(pose_handle_base_world_des_far.p, np.add(pose_handle_base_world_des_far.p, [0,0,1]), lifeTime=.5)
        for i in itertools.count():
            handle_base_ps.append(mech.get_pose_handle_base_world().p)
            if debug:
                p.addUserDebugLine(mech.get_pose_handle_base_world().p, np.add(mech.get_pose_handle_base_world().p, [0,0,1]), lifeTime=1)
            # keep fingers closed (doesn't seem to make a difference but should
            # probably continually close fingers)
            self._control_fingers('close', debug=debug)

            if self._at_des_handle_base_pose(pose_handle_base_world_des, q_offset, mech):
                finished = True
                break
            if i>timeout:
                if debug:
                    print('timeout limit reached. moving the next joint')
                break

            # get position error of the handle base, but velocity error of the
            # gripper COM
            p_handle_base_world_err, e_handle_base_world_err = self._get_pose_handle_base_world_error(pose_handle_base_world_des_far, q_offset, mech)
            v_tip_world_des = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            lin_v_com_world_err, omega_com_world_err = self._get_v_com_world_error(v_tip_world_des)

            # apply both position and velocity controls at the gripper COM
            f = np.multiply(self.k[0], p_handle_base_world_err) + np.multiply(self.d[0], lin_v_com_world_err)
            tau = np.multiply(self.k[1], e_handle_base_world_err) + np.multiply(self.d[1], omega_com_world_err)
            p_com_world, q_com_world = self._get_pose_com_('world')
            p.applyExternalForce(self._id, -1, f, p_com_world, p.WORLD_FRAME)
            # there is a bug in pyBullet. the link frame and world frame are inverted
            # this should be executed in the WORLD_FRAME
            p.applyExternalTorque(self._id, -1, tau, p.LINK_FRAME)
            p.stepSimulation()
        return handle_base_ps, finished

    def set_control_params(self, policy_type):
        if policy_type == 'Revolute':
            self.k = [2000.0,20.0]
        if policy_type == 'Prismatic':
            self.k = [800.0,20.0]

    def execute_trajectory(self, traj, mech, policy_type, debug):
        self.set_control_params(policy_type)

        # initial grasp pose
        pose_handle_world_init = util.Pose(*p.getLinkState(self._bb_id, mech.handle_id)[:2])
        p_tip_world_init = np.add(pose_handle_world_init.p, [0., .015, 0.]) # back up a little for better grasp
        pose_tip_world_init = util.Pose(p_tip_world_init, self.pose_tip_world_reset.q)

        # offset between the initial trajectory orientation and the initial handle orientation
        pose_handle_base_world = mech.get_pose_handle_base_world()
        q_offset = util.quat_math(traj[0].q, pose_handle_base_world.q, True, False)

        self._grasp_handle(pose_tip_world_init, debug)
        joint_motion = 0.0
        prev_pose_handle_base_world_des = None
        for (i, pose_handle_base_world_des) in enumerate(traj):
            start_p_handle_world = p.getLinkState(self._bb_id, mech.handle_id)[0]
            handle_base_ps, finished = self._move_PD(pose_handle_base_world_des, prev_pose_handle_base_world_des, q_offset, mech, debug)
            prev_pose_handle_base_world_des = pose_handle_base_world_des
            final_p_handle_world = p.getLinkState(self._bb_id, mech.handle_id)[0]
            joint_motion = np.add(joint_motion, np.linalg.norm(np.subtract(final_p_handle_world,start_p_handle_world)))
            if debug:
                for j in range(len(handle_base_ps)-1):
                    p.addUserDebugLine(handle_base_ps[j], handle_base_ps[j+1], [0,0,1])
            if not finished:
                break
        if self._in_contact(mech):
            pose_handle_world_final = util.Pose(*p.getLinkState(self._bb_id, mech.handle_id)[:2])
        else:
            pose_handle_world_final = None
        return joint_motion, pose_handle_world_final
