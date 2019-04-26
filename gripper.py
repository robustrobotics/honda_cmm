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
h - mechanism handle
d - door frame
'''
class Gripper:
    def __init__(self, bb_id, control_method):
        if control_method == 'PD':
            self.id = p.loadSDF("../models/gripper/gripper.sdf")[0]
        elif control_method == 'traj':
            self.id = p.loadSDF("../models/gripper/gripper_high_fric.sdf")[0]
        self.bb_id = bb_id
        self.left_finger_tip_id = 2
        self.right_finger_tip_id = 5
        self.left_finger_base_joint_id = 0
        self.right_finger_base_joint_id = 3
        self.p_b_t = [0.0002153, -0.02399915, -0.21146379]
        self.q_t_w_des =  [0.50019904,  0.50019904, -0.49980088, 0.49980088]
        self.finger_force = 5

    def set_tip_pose(self, p_t_w_des):
        p_b_w_des = util.transformation(self.p_b_t, p_t_w_des, self.q_t_w_des)
        p.resetBasePositionAndOrientation(self.id, p_b_w_des, self.q_t_w_des)
        p.stepSimulation()

    def grasp_handle(self, mechanism):
        p_t_w_init = [0., 0., .2]
        for t in range(10):
            self.set_tip_pose(p_t_w_init)

        for t in range(20):
            self.apply_command(util.Command(finger_state='open'), debug=False)

        p_h_w = p.getLinkState(self.bb_id, mechanism.handle_id)[0]
        for t in range(10):
            self.set_tip_pose(p_h_w)

        for t in range(200):
            self.apply_command(util.Command(finger_state='close'), debug=False)

    def apply_command(self, command, debug=False):
        # always control the fingers
        if command.finger_state == 'open':
            finger_angle = 0.2
        elif command.finger_state == 'close':
            finger_angle = 0.0

        p.setJointMotorControl2(self.id,self.left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,self.right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,2,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)
        p.setJointMotorControl2(self.id,5,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)

        # apply a force at the center of the gripper finger tips
        magnitude = 5.
        if command.force_dir is not None:
            p_b_w, q_b_w = p.getBasePositionAndOrientation(self.id)
            p_t_w = util.transformation(np.multiply(-1, self.p_b_t), p_b_w, q_b_w)
            force = np.multiply(magnitude, command.force_dir)
            p.applyExternalForce(self.id, -1, force, p_t_w, p.WORLD_FRAME)

            if debug:
                p.addUserDebugLine(p_t_w, np.add(p_t_w, force), lifeTime=.5)

        # control orientation with torque control
        if command.tip_orientation is not None:
            q_b_w = p.getBasePositionAndOrientation(self.id)[1]
            omega_b_w = p.getBaseVelocity(self.id)[1]
            q_t_w = q_b_w
            omega_t_w = omega_b_w

            q_t_w_err, _ = util.diff_quat(command.tip_orientation, q_t_w)
            e_t_w_err = util.euler_from_quaternion(q_t_w_err)
            omega_t_w_des = np.zeros(3)
            omega_t_w_err = omega_t_w_des - omega_t_w

            k = .006
            d = .001
            tau = np.dot(k, e_t_w_err) + np.dot(d, omega_t_w_err)
            # there is a bug in pyBullet. the link frame and world frame are inverted
            # this should be executed in the WORLD_FRAME
            p.applyExternalTorque(self.id, -1, tau, p.LINK_FRAME)

        if command.traj is not None:
            for (i, p_m_w_des) in enumerate(command.traj):
                if debug:
                    if i < len(command.traj)-1:
                        dir = np.subtract(command.traj[i+1], p_m_w_des)
                        end_point = np.multiply(1000., dir)
                        p.addUserDebugLine(p_m_w_des, np.add(p_m_w_des, end_point), lifeTime=.5)
                self.set_tip_pose(p_m_w_des)
                p.stepSimulation()

        p.stepSimulation()
