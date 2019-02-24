import pybullet as p
import numpy as np
from util import transformation

class Gripper:
    def __init__(self):
        self.id = p.loadSDF("models/gripper/gripper.sdf")[0]
        p.resetBasePositionAndOrientation(self.id,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
        self.left_finger_tip_id = 2
        self.right_finger_tip_id = 5
        self.left_finger_base_joint_id = 0
        self.right_finger_base_joint_id = 3
        self.pose_tip_base = self.get_pose_tip_base()

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

    def set_gripper_state(self, des_state):
        if des_state == 'open':
            finger_angle = 0.2
        elif des_state == 'close':
            finger_angle = 0.0
        else:
            print('desired gripper state can only be open or close')

        p.setJointMotorControl2(self.id,self.left_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=-finger_angle)#,force=self.fingerAForce)
        p.setJointMotorControl2(self.id,self.right_finger_base_joint_id,p.POSITION_CONTROL,targetPosition=finger_angle)#,force=self.fingerBForce)

        #p.setJointMotorControl2(self.kukaUid,10,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
        #p.setJointMotorControl2(self.kukaUid,13,p.POSITION_CONTROL,targetPosition=0,force=self.fingerTipForce)
        p.stepSimulation()
