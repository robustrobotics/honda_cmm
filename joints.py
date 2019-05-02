import numpy as np
import util
import pybullet as p

class JointModel(object):

    def get_pose_trajectory(self, bb_id, mechanism, delta_q):
        n_points = 100
        progress = np.linspace(0,1,n_points)
        p_m_w, q_m_w = p.getLinkState(bb_id, mechanism.handle_id)[:2]
        p_m_w, q_m_w = np.array(p_m_w), np.array(q_m_w)
        start_config = self.inverse_kinematics(p_m_w, np.array([0., 0., 0., 1.]))
        goal_config = start_config + delta_q

        positions = []
        for i in progress:
            config_new = start_config + delta_q * i
            next_pose_m_w = self.forward_kinematics(config_new)
            positions += [next_pose_m_w[0]]

        return util.Command(traj=positions)

class Prismatic(JointModel):

    def __init__(self, rigid_position, rigid_orientation, prismatic_dir):
        """

        :param rigid_position: A position along the prismatic joint.
        :param rigid_orientation: The orientation of the object.
        :param direction: A unit vector representing the direction of motion in the rigid frame.
        """

        self.rigid_position = np.array(rigid_position)
        self.rigid_orientation = np.array(rigid_orientation)
        self.prismatic_dir = np.array(prismatic_dir)

        # derived
        self.origin_M = util.pose_to_matrix(self.rigid_position, self.rigid_orientation)

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

    def get_force_direction(self, bb_id, mechanism):
        unit_vector = util.trans.unit_vector(self.prismatic_dir)
        command = util.Command(force_dir=unit_vector)
        return command

class Revolute(JointModel):

    def __init__(self, rot_center, rot_axis, rot_radius, rot_orientation):
        """

        :param rot_center: position for the center of rotation
        :param rot_axis: orientation for the center of rotation (fixed and assumed that
                            object rotates about the z axis)
        :param rot_radius: scalar representing the radius
        :param rot_orientation: orientation of the handle relative to the rotated frame
        """

        self.rot_center = np.array(rot_center)
        self.rot_axis = np.array(rot_axis)
        self.rot_radius = np.array(rot_radius)
        self.rot_orientation = np.array(rot_orientation)

        # derived
        self.center = util.pose_to_matrix(self.rot_center, self.rot_axis)
        self.radius = util.pose_to_matrix(self.rot_radius, self.rot_orientation)

    def forward_kinematics(self, q):
        rot_z = util.trans.rotation_matrix(-q,[0,0,1])
        M = util.trans.concatenate_matrices(self.center,rot_z,self.radius)
        p_z = M[:3,3]
        q_z = util.quaternion_from_matrix(M)
        return p_z, q_z

    def inverse_kinematics(self, p_z, q_z):
        z = util.pose_to_matrix(p_z, q_z)
        z_inv_c = np.dot(np.linalg.inv(z),self.center)
        inv_r = np.dot(np.linalg.inv(self.radius),z_inv_c)
        angle, direction, point = util.trans.rotation_from_matrix(inv_r)
        return angle

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
