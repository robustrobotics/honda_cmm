import numpy as np
import util
import pybullet as p

class Prismatic(object):

    def __init__(self, rigid_position, rigid_orientation, prismatic_dir):
        """

        :param rigid_position: A position along the prismatic joint.
        :param rigid_orientation: The orientation of the object.
        :param direction: A unit vector representing the direction of motion (in the world frame?).
        """

        self.rigid_position = np.array(rigid_position)
        self.rigid_orientation = np.array(rigid_orientation)
        self.prismatic_dir = np.array(prismatic_dir)

        # derived
        self.origin_M = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)

    def get_force_direction(self, bb_id, mechanism):
        unit_vector = util.trans.unit_vector(self.prismatic_dir)
        command = util.Command(force_dir=unit_vector)
        return command

    def get_pose_trajectory(self, bb_id, mechanism):
        pass

    def forward_kinematics(self, q):
        q_dir = np.multiply(q, self.prismatic_dir)
        q_dir = np.concatenate([q_dir, [1.]])
        p_z = np.dot(self.origin_M, q_dir)[:3]
        q_z = util.quaternion_from_matrix(self.origin_M)
        return p_z, q_z

    def inverse_kinematics(self, p_z, q_z):
        z_M = util.pose_to_matrix(p_z, q_z)
        inv_o_M = np.linalg.inv(self.origin_M)
        o_inv_z_M = np.dot(inv_o_M,z_M)
        trans = o_inv_z_M[:3,3]
        return np.dot(self.prismatic_dir, trans)

class Revolute(object):

    def __init__(self, rot_center, rot_axis, rot_radius, rot_orientation):
        """

        :param rot_center:
        :param rot_axis:
        :param rot_radius:
        :param rot_orientation:
        """

        self.rot_center = None
        self.rot_axis = None
        self.rot_radius = None
        self.rot_orientation = None

    def get_force_direction(self, bb_id, mechanism):
        delta_theta_mag = .001
        p_h_w, q_h_w = p.getLinkState(bb_id, mechanism.handle_id)[:2]
        p_d_w, q_d_w = p.getLinkState(bb_id, mechanism.door_base_id)[:2]
        p_h_d_w = np.subtract(p_h_w, p_d_w)
        R = np.linalg.norm(p_h_d_w[:2])

        # see which quadrant in, then calc theta
        if p_h_d_w[0] > 0 and p_h_d_w[1] > 0:
            theta = np.arcsin(p_h_d_w[1]/R)
        elif  p_h_d_w[0] < 0 and p_h_d_w[1] > 0:
            theta = np.pi - np.arcsin(p_h_d_w[1]/R)
        elif p_h_d_w[0] < 0 and p_h_d_w[1] < 0:
            theta = np.arcsin(p_h_d_w[1]/R) + np.pi
        elif p_h_d_w[0] > 0 and p_h_d_w[1] < 0:
            theta = 2*np.pi - np.arcsin(p_h_d_w[1]/R)

        # update to desired theta
        theta_new = theta - delta_theta_mag
        if mechanism.flipped:
            theta_new = theta + delta_theta_mag

        # calc new desired pose of handle
        p_h_d_w_des = [R*np.cos(theta_new), R*np.sin(theta_new), p_h_d_w[2]]
        p_h_w_des = np.add(p_d_w, p_h_d_w_des)
        direction = np.subtract(p_h_w_des, p_h_w)
        unit_vector = util.trans.unit_vector(direction)
        command = util.Command(force_dir=unit_vector)
        return command

    def get_pose_trajectory(self, bb_id, mechanism):
        pass

    def forward_kinematics(self):
        pass

    def inverse_kinematics(self):
        pass
