import pybullet as p
import numpy as np
import util

'''
# Naming convention
pose_ is a vector of length 7 representing a position + quaternion
p_ is a vector of length 3 representing a position
q_ is a vector of length 4 representing a quaternion
e_ is a vector of length 3 representing euler angles
the first following variable name represents the point/pose being described
the second following variable name indicates the frame that the point or pose is defined in

variable names:
w - world frame
b - gripper base frame
t - tip of the gripper frame
m - mechanism
'''
class Gripper:
    def __init__(self, bb_id):
        self.id = p.loadSDF("../models/gripper/gripper.sdf")[0]
        self.bb_id = bb_id
        self.left_finger_tip_id = 2
        self.right_finger_tip_id = 5
        self.left_finger_base_joint_id = 0
        self.right_finger_base_joint_id = 3
        self.pose_b_t = [0.0002153, -0.02399915, -0.21146379]
        self.q_t_w_des = [0.71, 0., 0., 0.71]
        self.finger_force = 5

    def set_tip_pose(self, pose_t_w_des):
        p_b_w_des = util.transformation(self.pose_b_t, pose_t_w_des[0], pose_t_w_des[1])
        q_b_w_des = pose_t_w_des[1]
        p.resetBasePositionAndOrientation(self.id, p_b_w_des, q_b_w_des)
        p.stepSimulation()

    def actuate_joint(self, mechanism):
        self.grasp_handle(mechanism)
        if mechanism.mechanism_type == 'Slider':
            self.actuate_prismatic(mechanism)
        '''
        elif mechanism.mechanism_type == 'Door':
            self.actuate_revolute(mechanism)
        '''

    def grasp_handle(self, mechanism):
        print('setting intial pose')
        pose_t_w_init = [[0., 0., .2], self.q_t_w_des]
        for t in range(10):
            self.set_tip_pose(pose_t_w_init)

        print('opening gripper')
        for t in range(20):
            self.apply_force(finger_state='open')

        print('moving gripper to mechanism')
        p_m_w = p.getLinkState(self.bb_id, mechanism.handle_id)[0]
        for t in range(10):
            self.set_tip_pose([p_m_w, self.q_t_w_des])

        print('closing gripper')
        for t in range(200):
            self.apply_force(finger_state='close')

    def actuate_prismatic(self, slider):
        print('moving slider')
        for t in range(500):
            axis_3d = [slider.axis[0], 0., slider.axis[1]]
            unit_vector = util.trans.unit_vector(axis_3d)
            magnitude = 5.
            self.apply_force(np.multiply(magnitude, unit_vector), q_t_w_des=self.q_t_w_des)
            p.stepSimulation()

    def apply_force(self, force=None, q_t_w_des=None, finger_state='close'):
        # apply a force at the center of the gripper finger tips
        if force is not None:
            p_b_w, q_b_w = p.getBasePositionAndOrientation(self.id)
            p_t_w = util.transformation(np.multiply(-1, self.pose_b_t), p_b_w, q_b_w)
            p.applyExternalForce(self.id, -1, force, p_t_w, p.WORLD_FRAME)

        # move fingers
        if finger_state == 'open':
            finger_angle = 0.2
        elif finger_state == 'close':
            finger_angle = 0.0

        p.setJointMotorControl2(self.id,self.left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,self.right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,2,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)
        p.setJointMotorControl2(self.id,5,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)

        # control orientation
        if q_t_w_des is not None:
            q_b_w = p.getBasePositionAndOrientation(self.id)[1]
            omega_b_w = p.getBaseVelocity(self.id)[1]
            q_t_w = q_b_w
            omega_t_w = omega_b_w

            q_t_w_err, _ = util.diff_quat(q_t_w_des, q_t_w)
            e_t_w_err = util.euler_from_quaternion(q_t_w_err)
            omega_t_w_des = np.zeros(3)
            omega_t_w_err = omega_t_w_des - omega_t_w

            k = .006
            d = .001
            tau = np.dot(k, e_t_w_err) + np.dot(d, omega_t_w_err)
            # there is a bug in pyBullet. the link frame and world frame are inverted
            # this should be executed in the WORLD_FRAME
            p.applyExternalTorque(self.id, -1, tau, p.LINK_FRAME)

        p.stepSimulation()
