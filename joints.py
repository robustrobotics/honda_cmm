class Prismatic(object):

    def __init__(self):
        """

        :param rigid_position: A position along the prismatic joint.
        :param rigid_orientation: The orientation of the object.
        :param direction: A unit vector representing the direction of motion (in the world frame?).
        """

        self.rigid_position = None
        self.rigid_orientation = None
        self.prismatic_dir = None

    def set_model(self, mechanism, bb_id):
        pass

    def forward_kinematics(self):
        pass

    def inverse_kinematics(self):
        pass

class Revolute(object):

    def __init__(self):
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

    def set_model(self, mechanism, bb_id):
        pass

    def forward_kinematics(self):
        pass

    def inverse_kinematics(self):
        pass
