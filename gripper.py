import pybullet as p
import numpy as np
import util
import time
from collections import namedtuple
import itertools
import sys
'''
# Naming convention
pose_ is a util.Pose()
p_ is a vector of length 3 representing a position
q_ is a vector of length 4 representing a quaternion
e_ is a vector of length 3 representing euler angles
the first following variable name represents the point/pose being described
the second following variable name indicates the frame that the point or pose is defined in

variable names:
w - world frame
b - gripper base frame
t - tip of the gripper frame
h - mechanism handle
d - door frame
com - center of mass of the entire gripper body
'''

class Gripper:
    def __init__(self, bb_id, k=[2000.,200.], d=[.45,.45]):
        self.id = p.loadSDF("../models/gripper/gripper_high_fric.sdf")[0]
        self.bb_id = bb_id
        self.left_finger_tip_id = 2
        self.right_finger_tip_id = 5
        self.left_finger_base_joint_id = 0
        self.right_finger_base_joint_id = 3
        self.q_t_w_des =  [0.50019904,  0.50019904, -0.49980088, 0.49980088]
        self.finger_force = 20
        self.k = k
        self.d = d

        # get mass of gripper
        mass = 0
        for link in range(p.getNumJoints(self.id)):
            mass += p.getDynamicsInfo(self.id, link)[0]
        self.mass = mass

    def get_p_tip_world(self):
        p_l_w = p.getLinkState(self.id, self.left_finger_tip_id)[0]
        p_r_w = p.getLinkState(self.id, self.right_finger_tip_id)[0]
        p_tip_w = np.mean([p_l_w, p_r_w], axis=0)
        return p_tip_w

    def get_p_tip_base(self):
        p_base_w, q_base_w = p.getBasePositionAndOrientation(self.id)
        p_tip_w = self.get_p_tip_world()
        p_tip_base = util.transformation(p_tip_w, p_base_w, q_base_w, inverse=True)
        return p_tip_base

    def set_tip_pose(self, pose_t_w_des, reset=False):
        p_b_t = np.multiply(-1, self.get_p_tip_base())
        p_b_w_des = util.transformation(p_b_t, pose_t_w_des.pos, pose_t_w_des.orn)
        p.resetBasePositionAndOrientation(self.id, p_b_w_des, pose_t_w_des.orn)
        p.stepSimulation()

    # control COM but monitor error in the task frame
    def get_pose_error(self, pose_t_w_des):
        p_t_w = self.get_p_tip_world()
        q_t_w = p.getBasePositionAndOrientation(self.id)[1]
        p_t_w_err = np.subtract(pose_t_w_des.pos, p_t_w)
        q_t_w_err = util.quat_math(pose_t_w_des.orn, q_t_w, False, True)
        e_t_w_err = util.euler_from_quaternion(q_t_w_err)
        return p_t_w_err, e_t_w_err

    def get_velocity_error(self, v_t_w_des):
        p_com_t, q_com_t = self.calc_COM('task')
        v_com_w_des = util.adjoint_transformation(v_t_w_des, p_com_t, q_com_t, inverse=True)

        v_b_w = np.concatenate(p.getBaseVelocity(self.id))
        p_com_b, q_com_b = self.calc_COM('base')
        v_com_w = util.adjoint_transformation(v_b_w, p_com_b, q_com_b, inverse=True)

        v_com_w_err = np.subtract(v_com_w_des, v_com_w)
        return v_com_w_err[:3], v_com_w_err[3:]

    def at_des_pose(self, pose_t_w_des):
        p_err_eps = .005
        #e_err_eps = .4
        p_com_w_err, e_com_w_err = self.get_pose_error(pose_t_w_des)
        return np.linalg.norm(p_com_w_err) < p_err_eps# and np.linalg.norm(e_com_w_err) < e_err_eps

    def move_PD(self, pose_t_w_des, debug=False, timeout=100):
        # move setpoint further away in a straight line between curr pose and goal pose
        add_dist = 0.1
        dir = np.subtract(pose_t_w_des.pos, self.get_p_tip_world())
        mag = np.linalg.norm([dir])
        unit_dir = np.divide(dir,mag)
        p_t_w_des_far = np.add(pose_t_w_des.pos,np.multiply(add_dist,unit_dir))
        pose_t_w_des_far = util.Pose(p_t_w_des_far, pose_t_w_des.orn)
        finished = False
        for i in itertools.count():
            # keep fingers closed (doesn't seem to make a difference but should
            # probably continually close fingers)
            self.control_fingers('close', debug=debug)
            if debug:
                p.addUserDebugLine(pose_t_w_des_far.pos, np.add(pose_t_w_des_far.pos,[0,0,10]), lifeTime=.5)
                p.addUserDebugLine(pose_t_w_des.pos, np.add(pose_t_w_des.pos,[0,0,10]), [1,0,0], lifeTime=.5)
                p_t_w = self.get_p_tip_world()
                p.addUserDebugLine(p_t_w, np.add(p_t_w,[0,0,10]), [0,1,0], lifeTime=.5)
                err = self.get_pose_error(pose_t_w_des)
                sys.stdout.write("\r%.3f %.3f" % (np.linalg.norm(err[0]), np.linalg.norm(err[1])))
                util.vis_frame(*pose_t_w_des)
            if self.at_des_pose(pose_t_w_des):
                finished = True
                break
            if i>timeout:
                if debug:
                    print('timeout limit reached. moving the next joint')
                break
            p_com_w_err, e_com_w_err = self.get_pose_error(pose_t_w_des_far)
            v_t_w_des = [0., 0., 0., 0., 0., 0.]
            lin_v_com_w_err, omega_com_w_err = self.get_velocity_error(v_t_w_des)

            f = np.multiply(self.k[0], p_com_w_err) + np.multiply(self.d[0], lin_v_com_w_err)
            tau = np.multiply(self.k[1], e_com_w_err) + np.multiply(self.d[1], omega_com_w_err)

            p_com_w, q_com_t = self.calc_COM('world')
            p.applyExternalForce(self.id, -1, f, p_com_w, p.WORLD_FRAME)
            # there is a bug in pyBullet. the link frame and world frame are inverted
            # this should be executed in the WORLD_FRAME
            p.applyExternalTorque(self.id, -1, tau, p.LINK_FRAME)
            p.stepSimulation()
        return finished

    def grasp_handle(self, pose_t_w_des, debug=False):
        # move to default start pose
        p_t_w_init = [0., 0., .2]
        q_t_w_init = [0.50019904,  0.50019904, -0.49980088, 0.49980088]
        pose_t_w_init = util.Pose(p_t_w_init, q_t_w_init)
        for t in range(10):
            self.set_tip_pose(pose_t_w_init, reset=True)

        # open fingers
        for t in range(10):
            self.control_fingers('open', debug=debug)

        # move to desired pose
        for t in range(10):
            self.set_tip_pose(pose_t_w_des, reset=True)

        # close fingers
        for t in range(10):
            self.control_fingers('close', debug=debug)

    def control_fingers(self, finger_state, debug=False):
        if finger_state == 'open':
            finger_angle = 0.2
        elif finger_state == 'close':
            finger_angle = 0.0
        p.setJointMotorControl2(self.id,self.left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,self.right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,2,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)
        p.setJointMotorControl2(self.id,5,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)
        p.stepSimulation()

    def execute_trajectory(self, grasp_pose, traj, debug=False, callback=None, bb=None, mech=None):
        self.grasp_handle(grasp_pose, debug)
        start_time = time.time()
        motion = 0
        for (i, pose_t_w_des) in enumerate(traj):
            if mech:
                start_mech_pose = p.getLinkState(self.bb_id, mech.handle_id)[0]
            finished = self.move_PD(pose_t_w_des, debug)
            if mech:
                final_mech_pose = p.getLinkState(self.id, mech.handle_id)[0]
                motion += np.linalg.norm(np.subtract(final_mech_pose,start_mech_pose))
            if not finished:
                break
            if not callback is None:
                callback(bb)
        duration = time.time() - start_time
        return i/len(traj), duration, motion

    def in_contact(self, mech):
        points = p.getContactPoints(self.id, self.bb_id, linkIndexB=mech.handle_id)
        if len(points)>0:
            return True
        return False

    def calc_COM(self, mode):
        com_num = np.array([0., 0., 0.])
        for link_index in range(p.getNumJoints(self.id)):
            link_com = p.getLinkState(self.id, link_index)[0]
            link_mass = p.getDynamicsInfo(self.id, link_index)[0]
            com_num = np.add(com_num, np.multiply(link_mass,link_com))
        p_com_w = np.divide(com_num, self.mass)

        p_b_w, q_b_w = p.getBasePositionAndOrientation(self.id)
        q_com_w = q_b_w

        if mode == 'world':
            return p_com_w, q_com_w
        elif mode == 'task':
            p_t_w = self.get_p_tip_world()
            p_com_t = util.transformation(p_com_w, p_t_w, q_b_w, inverse=True)
            q_com_t = np.array([0.,0.,0.,1.])
            return p_com_t, q_com_t
        elif mode == 'base':
            p_com_b = util.transformation(p_com_w, p_b_w, q_b_w, inverse=True)
            q_com_b = np.array([0.,0.,0.,1.])
            return p_com_b, q_com_b
