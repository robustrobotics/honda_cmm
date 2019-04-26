import numpy as np
import util

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

    def forward_kinematics(self):
        pass

    def inverse_kinematics(self):
        pass
