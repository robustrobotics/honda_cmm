import numpy as np
import odio_urdf as urdf
import argparse
import pybullet as p
import pybullet_data
import aabbtree as aabb
import cv2
from gripper import Gripper
from joints import Prismatic, Revolute

np.random.seed(0)

class Mechanism(object):
    def __init__(self, p_type):
        """
        This is an interface for each Mechanism type. Each Mechanism must implement
        the get_links, get_joints, and get_bounding_box methods.
        :param p_type: The type of mechanism of the parent class.
        """
        self.mechanism_type = p_type
        self._links = []
        self._joints = []

    def get_links(self):
        return self._links

    def get_joints(self):
        return self._joints

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

    def __init__(self, x_offset, z_offset, range, axis, color):
        """

        :param x_offset: The offset in the x-dimension from the busybox backboard.
        :param z_offset: The offset in the z-dimension from the busybox backboard.
        :param range: The total distance spanned by the prismatic joint.
        :param axis: A 2-d unit vector with directions in the x and z directions.
        :param color: An 3-tuple of rgb values.
        """
        super(Slider, self).__init__('Slider')

        name = Slider.n_sliders
        Slider.n_sliders += 1

        slider_handle_name = 'slider_{0}_handle'.format(name)
        handle = urdf.Link(slider_handle_name,
                           urdf.Inertial(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Mass(value=0.1),
                               urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                           ),
                           urdf.Collision(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=0.02, length=0.05)
                               )
                           ),
                           urdf.Visual(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=0.02, length=0.05)
                               ),
                               urdf.Material('slider_{0}_color'.format(name),
                                   urdf.Color(rgba=(color[0], color[1], color[2], 1.0))
                               )
                           ))

        track = urdf.Link('slider_{0}_track'.format(name),
                           urdf.Inertial(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Mass(value=0.1),
                               urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                           ),
                           urdf.Collision(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Box(size=(range, 0.005, 0.02))
                               )
                           ),
                           urdf.Visual(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Box(size=(range, 0.005, 0.02))
                               ),
                               urdf.Material('slider_track_color'.format(name),
                                   urdf.Color(rgba=(0.6, 0.6, 0.6, 1.0))
                               )
                           ))

        joint = urdf.Joint('slider_{0}_joint'.format(name),
                           urdf.Parent('back_link'),
                           urdf.Child('slider_{0}_handle'.format(name)),
                           urdf.Axis(xyz=(axis[0], axis[1], 0)),
                           urdf.Origin(xyz=(x_offset, 0.05, z_offset), rpy=(1.57, 0, 0)),
                           urdf.Limit(lower=-range/2.0, upper=range/2.0),
                           urdf.Dynamics(friction=1.0, damping=1.0),
                           type='prismatic')

        angle = np.arctan2(axis[1], axis[0])
        track_joint = urdf.Joint('slider_{0}_track_joint'.format(name),
                                 urdf.Parent('back_link'),
                                 urdf.Child('slider_{0}_track'.format(name)),
                                 urdf.Origin(xyz=(x_offset, 0.025, z_offset),
                                             rpy=(0, -angle, 0)),
                                 type='fixed')

        self._links.append(handle)
        self._joints.append(joint)
        self._links.append(track)
        self._joints.append(track_joint)

        self.handle_name = slider_handle_name
        self.handle_id = None
        self.origin = (x_offset, z_offset)
        self.range = range
        self.handle_radius = 0.025
        self.axis = axis
        self.joint_model = None

    def get_bounding_box(self):
        a = np.arctan2(self.axis[1], self.axis[0])

        z_min = self.origin[1] - np.sin(a)*self.range/2.0 - self.handle_radius
        z_max = self.origin[1] + np.sin(a)*self.range/2.0 + self.handle_radius

        x_min = self.origin[0] - np.abs(np.cos(a))*self.range/2.0 - self.handle_radius
        x_max = self.origin[0] + np.abs(np.cos(a))*self.range/2.0 + self.handle_radius

        return aabb.AABB([(x_min, x_max), (z_min, z_max)])

    def set_joint_model(self, bb_id):
        p_back_w = p.getLinkState(bb_id, 0)[0]
        rigid_position = np.add(p_back_w, [self.origin[0], .05, self.origin[1]])
        rigid_orientation = [0., 0., 0., 1.]
        prismatic_dir = [self.axis[0], 0., self.axis[1]]
        self.joint_model = Prismatic(rigid_position, rigid_orientation, prismatic_dir)

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
        color = (np.random.uniform(0, 1),
                 np.random.uniform(0, 1),
                 np.random.uniform(0, 1))
        return Slider(x_offset, z_offset, range, axis, color)


class Door(Mechanism):
    n_doors = 0

    def __init__(self, door_offset, door_size, handle_offset, flipped, color):
        super(Door, self).__init__('Door')
        name = Door.n_doors
        Door.n_doors += 1

        dir = 1.0
        if flipped: dir = -1.0

        door_base_name = 'door_{0}_base'.format(name)
        door = urdf.Link(door_base_name,
                         urdf.Inertial(
                              urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                              urdf.Mass(value=0.1),
                              urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                         ),
                         urdf.Collision(
                              urdf.Origin(xyz=(-dir*door_size[0]/2.0, 0, 0), rpy=(0, 0, 0)),
                              urdf.Geometry(
                                  urdf.Box(size=(door_size[0], 0.01, door_size[1]))
                              )
                         ),
                         urdf.Visual(
                             urdf.Origin(xyz=(-dir*door_size[0]/2.0, 0, 0), rpy=(0, 0, 0)),
                             urdf.Geometry(
                                 urdf.Box(size=(door_size[0], 0.01, door_size[1]))
                             ),
                             urdf.Material('door_{0}_color'.format(name),
                                           urdf.Color(rgba=(color[0], color[1], color[2], 1.0))
                             )
                         ))

        door_handle_name = 'door_{0}_handle'.format(name)
        door_handle = urdf.Link(door_handle_name,
                         urdf.Inertial(
                              urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                              urdf.Mass(value=0.1),
                              urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                         ),
                         urdf.Collision(
                              urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                              urdf.Geometry(
                                  urdf.Cylinder(radius=0.015, length=0.05)
                              )
                         ),
                         urdf.Visual(
                             urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                             urdf.Geometry(
                                 urdf.Cylinder(radius=0.015, length=0.05)
                             ),
                             urdf.Material('door_{0}_color'.format(name),
                                           urdf.Color(rgba=(color[0], color[1], color[2], 1.0))
                             )
                         ))

        if flipped: limit = urdf.Limit(lower=0, upper=1.57)
        else: limit = urdf.Limit(lower=-1.57, upper=0)

        door_joint = urdf.Joint('door_{0}_joint'.format(name),
                                urdf.Child('door_{0}_base'.format(name)),
                                urdf.Parent('back_link'),
                                urdf.Axis(xyz=(0, 0, 1)),
                                urdf.Origin(xyz=(door_offset[0], 0.03, door_offset[1]), rpy=(0, 0, 0)),
                                limit,
                                type='revolute')

        door_handle_joint = urdf.Joint('door_{0}_handle_joint'.format(name),
                                       urdf.Parent('door_{0}_base'.format(name)),
                                       urdf.Child('door_{0}_handle'.format(name)),
                                       urdf.Origin(xyz=(dir*(-door_size[0]+0.02), 0.025, handle_offset), rpy=(1.57, 0, 0)),
                                       type='fixed')

        self._links.append(door)
        self._joints.append(door_joint)
        self._links.append(door_handle)
        self._joints.append(door_handle_joint)

        self.handle_name = door_handle_name
        self.handle_id = None
        self.door_base_name = door_base_name
        self.door_base_id = None
        self.origin = door_offset
        self.door_size = door_size
        self.flipped = flipped
        self.joint_model = None

    def get_bounding_box(self):
        """ This method should return a bounding box of the 2-dimensional
        backboard in which spans all mechanism configurations. The bounding
        box coordinates should be relative to the center of the backboard.
        :return: aabb.AABB
        """
        z_min = self.origin[1] - self.door_size[1]/2.0
        z_max = self.origin[1] + self.door_size[1]/2.0

        if self.flipped:
            x_min = self.origin[0]
            x_max = self.origin[0] + self.door_size[0]
        else:
            x_min = self.origin[0] - self.door_size[0]
            x_max = self.origin[0]
        return aabb.AABB([(x_min, x_max), (z_min, z_max)])

    def set_joint_model(self, bb_id):
        rot_center, rot_axis = p.getLinkState(bb_id, self.door_base_id)[:2]
        p_handle_world, rot_orientation = p.getLinkState(bb_id, self.handle_id)[:2]
        rot_radius = np.linalg.norm(np.subtract(p_handle_world, rot_center)[:2])
        self.joint_model = Revolute(rot_center, rot_axis, rot_radius, rot_orientation)

    @staticmethod
    def random(width, height):
        door_offset = (np.random.uniform(-width/2.0, width/2.0),
                       np.random.uniform(-height/2.0, height/2.0))
        door_size = (np.random.uniform(0.05, 0.15),
                     np.random.uniform(0.05, 0.15))
        # 0.015 is the handle radius.
        handle_offset = np.random.uniform(-door_size[1]/2+0.015, door_size[1]/2-0.015)

        flipped = np.random.binomial(n=1, p=0.5)
        color = (np.random.uniform(0, 1),
                 np.random.uniform(0, 1),
                 np.random.uniform(0, 1))

        return Door(door_offset, door_size, handle_offset, flipped, color)


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

    def set_mechanism_ids(self, bb_id):
        num_joints = p.getNumJoints(bb_id)
        for joint_id in range(num_joints):
            joint_info = p.getJointInfo(bb_id, joint_id)
            link_name = joint_info[12]
            if 'handle' in link_name.decode("utf-8"):
                for mech in self._mechanisms:
                    if mech.handle_name == link_name.decode("utf-8"):
                        mech.handle_id = joint_info[0]
            elif 'door' in link_name.decode("utf-8") and 'base' in link_name.decode("utf-8"):
                for mech in self._mechanisms:
                    if mech.mechanism_type == 'Door':
                        if mech.door_base_name == link_name.decode("utf-8"):
                            mech.door_base_id = joint_info[0]

    def set_joint_models(self, bb_id):
        for mech in self._mechanisms:
            mech.set_joint_model(bb_id)

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

    def actuate_joints(self, bb_id, gripper, control_method):
        for mechanism in self._mechanisms:
            gripper.grasp_handle(mechanism)
            if control_method == 'PD' or mechanism.mechanism_type=='Door':
                for t in range(500):
                    command = mechanism.joint_model.get_force_direction(bb_id, mechanism)
                    gripper.apply_command(command)
            elif control_method == 'traj':
                delta_q = .1
                command = mechanism.joint_model.get_pose_trajectory(bb_id, mechanism, delta_q)
                gripper.apply_command(command)

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
    def generate_random_busybox(min_mech=2, max_mech=6, mech_types=[Slider, Door], n_tries=25):
        """
        :param min_mech: int, The minimum number of mechanisms to be included on the busybox.
        :param max_mech: int, The maximum number of classes to be included on the busybox.
        :param mechs: list, A list of the classes of mechanisms to choose from.
        :param n_tries: list, How many attempts to generate a mechanism that does not overlap existing mechanisms.
        :return:
        """
        # Sample busybox dimensions.
        width = np.random.uniform(0.4, 0.8)
        height = np.random.uniform(0.2, 0.4)

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

        return BusyBox(width, height, mechs)


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', default=True)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--actuate', action='store_true')
    parser.add_argument('--control-method', default='PD')
    args = parser.parse_args()

    for ix in range(args.n):
        print('Busybox Number:',ix)
        bb = BusyBox.generate_random_busybox()

        if args.save:
            with open('models/busybox_{0}.urdf'.format(ix), 'w') as handle:
                handle.write(bb.get_urdf())

        if args.viz or args.images:
            client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            #p.setGravity(0, 0, -10)
            p.setRealTimeSimulation(1)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=180,
                cameraPitch=-30,
                cameraTargetPosition=(0., 0., 0.))
            plane_id = p.loadURDF("plane.urdf")

            with open('.busybox.urdf', 'w') as handle:
                handle.write(bb.get_urdf())
            model = p.loadURDF('.busybox.urdf', [0., -.3, 0.])
            bb.set_mechanism_ids(model)
            bb.set_joint_models(model)
            maxForce = 10
            mode = p.VELOCITY_CONTROL
            for jx in range(0, p.getNumJoints(model)):
                p.setJointMotorControl2(bodyUniqueId=model,
                                        jointIndex=jx,
                                        controlMode=mode,
                                        force=maxForce)

            if args.images:
                view_matrix = p.computeViewMatrix(cameraEyePosition=(-0.1, 0.1, 0.1),
                                                  cameraTargetPosition=(0, 0, 0),
                                                  cameraUpVector=(0, 0, 1))
                h, w, rgb, depth, seg = p.getCameraImage(200, 200, renderer=p.ER_TINY_RENDERER)
                img = np.array(rgb, dtype='uint8').reshape(200, 200, 4)
                # PyBullet has RGB but opencv user BGR.
                tmp_red = img[:, :, 0].tolist()
                img[:, :, 0] = img[:, :, 2]
                img[:, :, 2] = np.array(tmp_red)
                img = img[:, :, :3]

                cv2.imwrite('images/busybox_{0}.png'.format(ix), img)

            elif args.viz:
                if args.actuate:
                    gripper = Gripper(model)
                    bb.actuate_joints(model, gripper, args.control_method)
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    print('Exiting...')

            p.disconnect(client)
