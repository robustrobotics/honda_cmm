import pybullet as p
import numpy as np
from util import transformation

class Gripper:
    def __init__(self):
        self.id = p.loadSDF("models/gripper/gripper.sdf")[0]
        self.left_finger_tip_id = 2
        self.right_finger_tip_id = 5
        self.left_finger_base_joint_id = 0
        self.right_finger_base_joint_id = 3
        self.pose_tip_base = self.get_pose_tip_base()
        self.finger_force = 15

    def get_pose_tip_base(self):
        p_base_w, orn_base_w = p.getBasePositionAndOrientation(self.id)
        p_l_w = p.getLinkState(self.id, self.left_finger_tip_id)[0]
        p_r_w = p.getLinkState(self.id, self.right_finger_tip_id)[0]
        p_tip_base = np.mean([p_l_w, p_r_w], axis=0) - p.getBasePositionAndOrientation(self.id)[0]
        orn_tip_base = [0., 0., 0., 1.]
        return p_tip_base, orn_tip_base

    def set_tip_pose(self, pose_tip_w_des):
        p_base_tip = np.multiply(-1, self.pose_tip_base[0])
        p_base_world_des = transformation(p_base_tip, pose_tip_w_des[0], pose_tip_w_des[1])
        p.resetBasePositionAndOrientation(self.id, p_base_world_des, pose_tip_w_des[1])
        p.stepSimulation()

    def apply_force(self, force=None, finger_state='close'):
        if force is not None:
            # apply a force at the center of the gripper finger tips
            p_tip_base = self.pose_tip_base[0]
            p_base_w, orn_base_w = p.getBasePositionAndOrientation(self.id)
            p_tip_w = transformation(p_tip_base, p_base_w, orn_base_w)
            p.applyExternalForce(self.id, -1, force, p_tip_w, p.WORLD_FRAME)

        if finger_state == 'open':
            finger_angle = 0.2
        elif finger_state == 'close':
            finger_angle = 0.0

        p.setJointMotorControl2(self.id,self.left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,self.right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle,force=self.finger_force)
        p.setJointMotorControl2(self.id,2,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)
        p.setJointMotorControl2(self.id,5,p.POSITION_CONTROL,targetPosition=0,force=self.finger_force)

        p.stepSimulation()
