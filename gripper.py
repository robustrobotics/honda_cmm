import pybullet as p
import numpy as np
import util
import time
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
    def __init__(self, bb_id, control_method):
        if control_method == 'force':
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
        self.finger_force = 20

        # get mass of gripper
        mass = 0
        for link in range(p.getNumJoints(self.id)):
            mass += p.getDynamicsInfo(self.id, link)[0]
        self.mass = mass

    def set_tip_pose(self, pose_t_w_des, reset=False):
        p_b_w_des = util.transformation(self.p_b_t, pose_t_w_des.pos, pose_t_w_des.orn)
        # move back just a little bit
        p_b_w_des[1] += .005
        p.resetBasePositionAndOrientation(self.id, p_b_w_des, pose_t_w_des.orn)
        p.stepSimulation()

    def get_pose_error(self, pose_t_w_des):
        p_com_w, q_com_w, p_com_t = self.calc_COM('task')
        p_com_w_des = util.transformation(p_com_t, pose_t_w_des.pos, pose_t_w_des.orn)
        p_com_w_err = np.subtract(p_com_w_des, p_com_w)

        q_com_w_err, _ = util.diff_quat(pose_t_w_des.orn, q_com_w)
        e_com_w_err = util.euler_from_quaternion(q_com_w_err)
        return p_com_w_err, e_com_w_err

    def at_des_pose(self, pose_t_w_des):
        p_err_eps = .01
        e_err_eps = .001
        p_com_w_err, e_com_w_err = self.get_pose_error(pose_t_w_des)
        return all(p < p_err_eps for p in p_com_w_err) and all(e < e_err_eps for e in e_com_w_err)

    # TODO: actually just P control at the moment. need to add D control
    def move_PD(self, pose_t_w_des):
        k=[20, 2]
        d=[800., .9]
        while not self.at_des_pose(pose_t_w_des):
            p_com_w_err, e_com_w_err = self.get_pose_error(pose_t_w_des)
            '''
            v_b_w = np.concatenate(p.getBaseVelocity(self.id))
            v_t_w = util.adjoint_transformation(v_b_w, self.p_b_t, [0., 0., 0., 1.])
            v_t_w_err = -1*v_t_w
            '''
            f = np.dot(k[0], p_com_w_err) #+ np.dot(d[0], v_t_w_err[:3])
            tau = np.dot(k[1], e_com_w_err) #+ np.dot(d[1], v_t_w_err[3:])

            p_com_w, _ = self.calc_COM('world')
            p.applyExternalForce(self.id, -1, f, p_com_w, p.WORLD_FRAME)
            # there is a bug in pyBullet. the link frame and world frame are inverted
            # this should be executed in the WORLD_FRAME
            p.applyExternalTorque(self.id, -1, tau, p.LINK_FRAME)
            p.stepSimulation()

    def grasp_handle(self, pose_t_w_des, viz=False):
        # default values for moving the gripper to a pose before grasping handle
        p_t_w_init = [0., 0., .2]
        q_t_w_init = [0.50019904,  0.50019904, -0.49980088, 0.49980088]
        pose_t_w_init = util.Pose(p_t_w_init, q_t_w_init)
        for t in range(10):
            self.set_tip_pose(pose_t_w_init, reset=True)

        for t in range(10):
            self.apply_command(util.Command(finger_state='open'), debug=False, viz=viz)

        for t in range(10):
            self.set_tip_pose(pose_t_w_des, reset=True)

        for t in range(10):
            self.apply_command(util.Command(finger_state='close'), debug=False, viz=viz)

    def apply_command(self, command, debug=False, viz=False, callback=None, bb=None):
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
            # gravity compensation
            g_force = self.mass*10.
            force_w_grav = np.add(force, [0., 0., g_force])
            p.applyExternalForce(self.id, -1, force_w_grav, p_t_w, p.WORLD_FRAME)

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
            for (i, pose_t_w_des) in enumerate(command.traj):
                self.move_PD(pose_t_w_des)
                if not callback is None:
                    callback(bb)

        p.stepSimulation()

    def calc_COM(self, mode='world'):
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
            p_t_w = util.transformation(np.multiply(-1, self.p_b_t),  p_b_w, q_b_w)
            p_com_t = util.transformation(p_com_w, p_t_w, q_b_w, inverse=True)
            return p_com_w, q_com_w, p_com_t
