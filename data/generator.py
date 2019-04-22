import numpy as np
import odio_urdf as urdf
import argparse
import pybullet as p
import pybullet_data


class Mechanism(object):
    def __init__(self, p_type):
        """
        This is an interface for each Mechanism type. Each Mechanism must implement
        the getURDF and getBoundingBox methods.
        :param p_type: The type of mechanism of the parent class.
        """
        self.mechanism_type = p_type

    def get_urdf(self):
        raise NotImplementedError('URDF generation not implemented for mechanism: {0}'.format(self.p_type))

    def get_bounding_box(self):
        raise NotImplementedError('Collision Bounding Box not implemented for mechanism: {0}'.format(self.p_type))


class Slider(Mechanism):
    def __init__(self):
        super(Slider, self).__init__('Slider')

    def get_urdf(self):
        pass

    def get_bounding_box(self):
        pass


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
                                 urdf.Origin(xyz=(0, 0, 0.2), rpy=(0, 0, 0)),
                                 type='fixed')

        self._links.append(base_link)
        self._links.append(back_link)
        self._joints.append(fixed_joint)

    def get_urdf(self):
        """
        :return: A str representation of the busybox's urdf.
        """
        robot = urdf.Robot('busybox', *(self._links+self._joints))
        return str(robot)


    @staticmethod
    def generate_random_busybox(min_mech=2, max_mech=4, mechs=[Slider]):
        """
        :param min_mech: int, The minimum number of mechanisms to be included on the busybox.
        :param max_mech: int, The maximum number of classes to be included on the busybox.
        :param mechs: list, A list of the classes of mechanisms to choose from.
        :return:
        """
        bb = BusyBox(0.6, 0.3, [])

        # TODO: Pick BusyBox backboard size.
        # TODO: Pick number of mechanisms.
        n_mech = np.random.randint(low=min_mech, high=max_mech+1)
        for _ in range(n_mech):
            # TODO: For each mechanism pick the type.
            mech_class = np.random.choice(mechs)
            mech = mech_class()

            # TODO: Check for collisions.

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
        p.setRealTimeSimulation(0)
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
