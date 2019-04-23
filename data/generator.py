import numpy as np
import odio_urdf as urdf
import argparse
import pybullet as p
import pybullet_data
import aabbtree as aabb


class Mechanism(object):
    def __init__(self, p_type):
        """
        This is an interface for each Mechanism type. Each Mechanism must implement
        the get_links, get_joints, and get_bounding_box methods.
        :param p_type: The type of mechanism of the parent class.
        """
        self.mechanism_type = p_type

    def get_links(self):
        raise NotImplementedError('get_links not implemented for mechanism: {0}'.format(self.mechanism_type))

    def get_joints(self):
        raise NotImplementedError('get_joints not implemented for mechanism: {0}'.format(self.mechanism_type))

    def get_bounding_box(self):
        """ This method should return a bounding box of the 2-dimensional
        backboard in which spans all mechanism configurations. The bounding
        box coordinates should be relative to the center of the backboard.
        :return: aabb.AABB
        """
        raise NotImplementedError('Collision Bounding Box not implemented for mechanism: {0}'.format(self.mechanism_type))

    @staticmethod
    def random():
        raise NotImplementedError('Cannot generate a random mechanism.')


class Slider(Mechanism):
    n_sliders = 0

    def __init__(self, x_offset, z_offset, range, axis):
        """

        :param x_offset: The offset in the x-dimension from the busybox backboard.
        :param z_offset: The offset in the z-dimension from the busybox backboard.
        :param range: The total distance spanned by the prismatic joint.
        :param axis: A 2-d unit vector with directions in the x and z directions.
        """
        super(Slider, self).__init__('Slider')
        self._links = []
        self._joints = []

        name = Slider.n_sliders
        Slider.n_sliders += 1

        handle = urdf.Link('slider_{0}'.format(name),
                           urdf.Inertial(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Mass(value=0.1),
                               urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                           ),
                           urdf.Collision(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=0.025, length=0.1)
                               )
                           ),
                           urdf.Visual(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=0.025, length=0.1)
                               ),
                               urdf.Material('orange',
                                             urdf.Color(rgba=(1.0, 0.6, 0, 1.0))
                               )
                           ))

        joint = urdf.Joint('slider_{0}_joint'.format(name),
                           urdf.Parent('back_link'),
                           urdf.Child('slider_{0}'.format(name)),
                           urdf.Axis(xyz=(axis[0], axis[1], 0)),
                           urdf.Origin(xyz=(x_offset, 0.075, z_offset), rpy=(1.57, 0, 0)),
                           urdf.Limit(lower=-range/2.0, upper=range/2.0),
                           type='prismatic')

        self._links.append(handle)
        self._joints.append(joint)

        self.origin = (x_offset, z_offset)
        self.range = range
        self.handle_radius = 0.025
        self.axis = axis

    def get_links(self):
        return self._links

    def get_joints(self):
        return self._joints

    def get_bounding_box(self):
        a = np.arctan2(self.axis[1], self.axis[0])

        z_min = self.origin[1] - np.sin(a)*self.range/2.0 - self.handle_radius
        z_max = self.origin[1] + np.sin(a)*self.range/2.0 + self.handle_radius

        x_min = self.origin[0] - np.abs(np.cos(a))*self.range/2.0 - self.handle_radius
        x_max = self.origin[0] + np.abs(np.cos(a))*self.range/2.0 + self.handle_radius
        
        return aabb.AABB([(x_min, x_max), (z_min, z_max)])


    @staticmethod
    def random(width, height):
        """
        Generate a random slider within the busybox of dimensions width x height.
        :param width: float, width of the busybox.
        :param height: float, height of the busybox.
        :return: Slider object.
        """
        x_offset = np.random.uniform(-width/2.0, width/2.0)
        z_offset = np.random.uniform(-height/2.0, height/2.0)
        range = np.random.uniform(0.1, 0.5)
        angle = np.random.uniform(0, np.pi)
        axis = (np.cos(angle), np.sin(angle))

        slider = Slider(x_offset, z_offset, range, axis)
        return slider


class BusyBox(object):
    def __init__(self, width, height, mechanisms):
        self._mechanisms = mechanisms
        self._links = []
        self._joints = []
        self._create_skeleton(width, height)

    def _create_skeleton(self, width, height):
        """
        The busybox skeleton consists of a base and backboard joined by
        a fixed Joint. Note all unspecified dimensions are fixed.
        :param width: The length of the busybox in the x-dimension.
        :param height: The height of the busybox in the z-dimenstion.
        """
        base_link = urdf.Link('base_link',
                              urdf.Inertial(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Mass(value=0),
                                  urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                              ),
                              urdf.Collision(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Geometry(
                                      urdf.Box(size=(width, 0.3, 0.1))
                                  )
                              ),
                              urdf.Visual(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Geometry(
                                      urdf.Box(size=(width, 0.3, 0.1))
                                  ),
                                  urdf.Material('brown',
                                                urdf.Color(rgba=(0.82, 0.71, 0.55, 1.0))
                                  )
                              ))

        back_link = urdf.Link('back_link',
                              urdf.Inertial(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Mass(value=0.5),
                                  urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                              ),
                              urdf.Collision(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Geometry(
                                      urdf.Box(size=(width, 0.05, height))
                                  )
                              ),
                              urdf.Visual(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Geometry(
                                      urdf.Box(size=(width, 0.05, height))
                                  ),
                                  urdf.Material('brown',
                                                urdf.Color(rgba=(0.82, 0.71, 0.55, 1.0))
                                                )
                              ))

        fixed_joint = urdf.Joint('fixed_backboard',
                                 urdf.Parent('base_link'),
                                 urdf.Child('back_link'),
                                 urdf.Origin(xyz=(0, 0, height/2.0 + 0.05), rpy=(0, 0, 0)),
                                 type='fixed')

        self._links.append(base_link)
        self._links.append(back_link)
        self._joints.append(fixed_joint)

    def get_urdf(self):
        """
        :return: A str representation of the busybox's urdf.
        """
        elements = self._links + self._joints
        for m in self._mechanisms:
            elements += m.get_links()
            elements += m.get_joints()
        robot = urdf.Robot('busybox', *elements)
        return str(robot)


    @staticmethod
    def _check_collision(width, height, mechs, mech):
        tree = aabb.AABBTree()
        # Add edge bounding boxes for backboard.
        tree.add(aabb.AABB([(-width/2.0, width/2.0), (height/2.0, height/2.0+1)]))  # top
        tree.add(aabb.AABB([(-width/2.0, width/2.0), (-height/2.0-1, -height/2.0)]))  # bottom
        tree.add(aabb.AABB([(-width/2.0-1, -width/2.0), (-height/2.0, height/2.0)]))  # left
        tree.add(aabb.AABB([(width/2.0, width/2.0+1), (-height/2.0, height/2.0)]))  # right

        # Get the bounding box for each existing mechanism.
        for ix, m in enumerate(mechs):
            tree.add(m.get_bounding_box(), str(ix))

        # Get the bounding box of the current mechanism and check overlap.
        if tree.does_overlap(mech.get_bounding_box()):
            return True
        else:
            return False

    @staticmethod
    def generate_random_busybox(min_mech=2, max_mech=4, mech_types=[Slider], n_tries=10):
        """
        :param min_mech: int, The minimum number of mechanisms to be included on the busybox.
        :param max_mech: int, The maximum number of classes to be included on the busybox.
        :param mechs: list, A list of the classes of mechanisms to choose from.
        :param n_tries: list, How many attempts to generate a mechanism that does not overlap existing mechanisms.
        :return:
        """
        # Sample busybox dimensions.
        width = np.random.uniform(0.4, 0.8)
        height = np.random.uniform(0.2, 0.6)

        # Sample number of mechanisms.
        mechs = []
        n_mech = np.random.randint(low=min_mech, high=max_mech+1)
        for _ in range(n_mech):
            # For each mechanism pick the type.
            mech_class = np.random.choice(mech_types)
            for _ in range(n_tries):
                mech = mech_class.random(width, height)
                # Check for collisions.
                if not BusyBox._check_collision(width, height, mechs, mech):
                    mechs.append(mech)
                    break

        bb = BusyBox(width, height, mechs)
        return bb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    bb = BusyBox.generate_random_busybox()
    print(bb.get_urdf())

    if args.viz:
        client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=210,
            cameraPitch=-52,
            cameraTargetPosition=(0., 0., 0.))
        plane_id = p.loadURDF("plane.urdf")

        with open('.busybox.urdf', 'w') as handle:
            handle.write(bb.get_urdf())
        model = p.loadURDF('.busybox.urdf')

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print('Exiting...')
