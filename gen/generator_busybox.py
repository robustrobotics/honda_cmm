import numpy as np
import odio_urdf as urdf
import argparse
import pybullet as p
import pybullet_data
import aabbtree as aabb
import cv2
from actions.gripper import Gripper
from utils import util
from collections import namedtuple

MechanismParams = namedtuple('MechanismParams', 'type params')
SliderParams = namedtuple('SliderParams', 'x_offset z_offset range axis')
DoorParams = namedtuple('DoorParams', 'door_offset door_size handle_offset_z flipped')

class Mechanism(object):
    def __init__(self, p_type):
        """
        This is an interface for each Mechanism type. Each Mechanism must implement
        the get_links, get_joints, and get_bounding_box methods.
        :param p_type: The type of mechanism of the parent class.
        """
        self.mechanism_type = p_type
        self.handle_length = 0.05
        self._handle_id = None
        self._bb = None
        self._links = []
        self._joints = []

    def _get_bb_id(self):
        return self._bb.get_bb_id()

    def _get_handle_id(self):
        assert self._handle_id is not None, 'BusyBox.set_mechanism_ids() must be called to access pyBullet ids'
        return self._handle_id

    def get_links(self):
        return self._links

    def get_joints(self):
        return self._joints

    def get_pose_handle_base_world(self):
        handle_id = self._get_handle_id()
        bb_id = self._get_bb_id()
        pose_handle_world = util.Pose(*p.getLinkState(bb_id, handle_id)[:2])
        p_handle_base = [0., 0., self.handle_length/2]
        p_handle_base_world = util.transformation(p_handle_base, *pose_handle_world)
        return util.Pose(p_handle_base_world, pose_handle_world.q)

    def get_bounding_box(self):
        """ This method should return a bounding box of the 2-dimensional
        backboard in which spans all mechanism configurations. The bounding
        box coordinates should be relative to the center of the backboard.
        :return: aabb.AABB
        """
        raise NotImplementedError('Collision Bounding Box not implemented for mechanism: {0}'.format(self.mechanism_type))

    def get_mechanism_tuple(self):
        raise NotImplementedError('get_mechanism_tuple not implemented for mechanism: {0}'.format(self.mechanism_type))

    def get_handle_pose(self):
        handle_id = self._get_handle_id()
        bb_id = self._get_bb_id()
        return util.Pose(*p.getLinkState(bb_id, handle_id)[:2])

    def get_contact_points(self, gripper_id):
        handle_id = self._get_handle_id()
        bb_id = self._get_bb_id()
        return p.getContactPoints(gripper_id, bb_id, linkIndexB=handle_id)

    @staticmethod
    def mech_from_mech_params(mech_params):
        color = (1., 0., 0.)
        if mech_params.type == 'Slider':
            return Slider(*mech_params.params, color)
        if mech_params.type == 'Door':
            return Door(*mech_params.params, color)

    @staticmethod
    def random():
        raise NotImplementedError('Cannot generate a random mechanism.')


class Slider(Mechanism):
    n_sliders = 0

    def __init__(self, x_offset, z_offset, range, axis, color, bb_thickness=0.05):
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

        handle_radius = 0.02
        slider_handle_name = 'slider_{0}_handle'.format(name)
        slider_track_name = 'slider_{0}_track'.format(name)
        handle = urdf.Link(slider_handle_name,
                           urdf.Inertial(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Mass(value=0.1),
                               urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                           ),
                           urdf.Collision(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=handle_radius, length=self.handle_length)
                               )
                           ),
                           urdf.Visual(
                               urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                               urdf.Geometry(
                                   urdf.Cylinder(radius=handle_radius, length=self.handle_length)
                               ),
                               urdf.Material('slider_{0}_color'.format(name),
                                   urdf.Color(rgba=(color[0], color[1], color[2], 1.0))
                               )
                           ))

        track = urdf.Link(slider_track_name.format(name),
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
                           urdf.Origin(xyz=(x_offset, self.handle_length, z_offset), rpy=(1.57, 0, 0)),
                           urdf.Limit(lower=-range/2.0, upper=range/2.0),
                           urdf.Dynamics(friction=1.0, damping=1.0),
                           type='prismatic')

        angle = -np.arctan2(axis[1], axis[0])
        track_joint = urdf.Joint('slider_{0}_track_joint'.format(name),
                                 urdf.Parent('back_link'),
                                 urdf.Child('slider_{0}_track'.format(name)),
                                 urdf.Origin(xyz=(x_offset, bb_thickness/2, z_offset),
                                             rpy=(0, angle, 0)),
                                 type='fixed')

        self._links.append(handle)
        self._joints.append(joint)
        self._links.append(track)
        self._joints.append(track_joint)

        self.handle_name = slider_handle_name
        self.track_name = slider_track_name
        self.origin = (x_offset, z_offset)
        self.range = range
        self.handle_radius = handle_radius
        self.axis = axis

    def get_bounding_box(self):
        a = np.arctan2(self.axis[1], self.axis[0])

        z_min = self.origin[1] - np.sin(a)*self.range/2.0 - self.handle_radius
        z_max = self.origin[1] + np.sin(a)*self.range/2.0 + self.handle_radius

        x_min = self.origin[0] - np.abs(np.cos(a))*self.range/2.0 - self.handle_radius
        x_max = self.origin[0] + np.abs(np.cos(a))*self.range/2.0 + self.handle_radius

        return aabb.AABB([(x_min, x_max), (z_min, z_max)])

    def get_mechanism_tuple(self):
        return MechanismParams(self.mechanism_type,
                SliderParams(self.origin[0],
                                self.origin[1],
                                self.range,
                                self.axis))

    def get_max_net_motion(self):
        return self.range/2

    def reset(self):
        handle_id = self._get_handle_id()
        bb_id = self._get_bb_id()
        p.resetJointState(bb_id, handle_id, 0.0)

    @staticmethod
    def random(width, height, bb_thickness=0.05):
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
        color = (1, 0, 0)
        return Slider(x_offset, z_offset, range, axis, color, bb_thickness)

class Door(Mechanism):
    n_doors = 0

    def __init__(self, door_offset, door_size, handle_offset_z, flipped, color, bb_thickness=0.05):
        super(Door, self).__init__('Door')
        name = Door.n_doors
        Door.n_doors += 1

        dir = 1.0
        if flipped: dir = -1.0

        thickness = 0.01
        handle_radius = 0.015
        handle_offset_x = 0.005
        door_base_name = 'door_{0}_base'.format(name)
        door = urdf.Link(door_base_name,
                         urdf.Inertial(
                              urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                              urdf.Mass(value=0.1),
                              urdf.Inertia(ixx=0.001, ixy=0, ixz=0, iyy=0.001, iyz=0, izz=0.001)
                         ),
                         urdf.Collision(
                              urdf.Origin(xyz=(-dir*door_size[0]/2.0, thickness/2, 0), rpy=(0, 0, 0)),
                              urdf.Geometry(
                                  urdf.Box(size=(door_size[0], thickness, door_size[1]))
                              )
                         ),
                         urdf.Visual(
                             urdf.Origin(xyz=(-dir*door_size[0]/2.0, thickness/2, 0), rpy=(0, 0, 0)),
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
                                  urdf.Cylinder(radius=handle_radius, length=self.handle_length)
                              )
                         ),
                         urdf.Visual(
                             urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                             urdf.Geometry(
                                 urdf.Cylinder(radius=handle_radius, length=self.handle_length)
                             ),
                             urdf.Material('door_{0}_handle_color'.format(name),
                                           urdf.Color(rgba=(color[0]*0.5, color[1]*0.5, color[2]*0.5, 1.0))
                             )
                         ))

        if flipped: limit = urdf.Limit(lower=0, upper=2.355)
        else: limit = urdf.Limit(lower=-2.355, upper=0)

        door_joint = urdf.Joint('door_{0}_joint'.format(name),
                                urdf.Child('door_{0}_base'.format(name)),
                                urdf.Parent('back_link'),
                                urdf.Axis(xyz=(0, 0, 1)),
                                urdf.Origin(xyz=(door_offset[0], bb_thickness/2, door_offset[1]), rpy=(0, 0, 0)),
                                limit,
                                type='revolute')

        door_handle_joint = urdf.Joint('door_{0}_handle_joint'.format(name),
                                       urdf.Parent('door_{0}_base'.format(name)),
                                       urdf.Child('door_{0}_handle'.format(name)),
                                       urdf.Origin(xyz=(dir*(-door_size[0]+handle_radius+handle_offset_x), \
                                                        self.handle_length/2,
                                                        handle_offset_z),
                                                    rpy=(1.57, 0, 0)),
                                       type='fixed')

        self._links.append(door)
        self._joints.append(door_joint)
        self._links.append(door_handle)
        self._joints.append(door_handle_joint)
        self._door_base_id = None

        self.handle_name = door_handle_name
        self.door_base_name = door_base_name
        self.origin = door_offset
        self.door_size = door_size
        self.handle_offset_z = handle_offset_z
        self.handle_offset_x = handle_offset_x
        self.handle_radius = handle_radius
        self.flipped = flipped

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

    def get_mechanism_tuple(self):
        return MechanismParams(self.mechanism_type,
                DoorParams(self.origin,
                            self.door_size,
                            self.handle_offset_z,
                            self.flipped))

    def _get_door_base_id(self):
        assert self._door_base_id is not None, 'BusyBox.set_mechanism_ids() must be called to access pyBullet ids'
        return self._door_base_id

    def get_rot_center(self):
        bb_id = self._get_bb_id()
        door_base_id = self._get_door_base_id()
        return p.getLinkState(bb_id, door_base_id)[0]

    def get_max_net_motion(self):
        motion_radius = self.door_size[0] - (self.handle_radius + self.handle_offset_x)
        return np.sqrt(2*motion_radius**2)

    def get_radius_x(self):
        p_handle_base_world = self.get_pose_handle_base_world().p
        p_rot_center_world_true = self.get_rot_center()
        radius_x = abs(np.subtract(p_handle_base_world, p_rot_center_world_true)[0])
        return radius_x

    def reset(self):
        door_base_id = self._get_door_base_id()
        bb_id = self._get_bb_id()
        p.resetJointState(bb_id, door_base_id, 0.0)

    @staticmethod
    def random(width, height, bb_thickness=0.05):
        door_offset = (np.random.uniform(-width/2.0, width/2.0),
                       np.random.uniform(-height/2.0, height/2.0))
        door_size = (np.random.uniform(0.08, 0.15),
                     np.random.uniform(0.05, 0.15))
        # 0.015 is the handle radius.
        # offset in the z (up/down) direction
        handle_offset_z = np.random.uniform(-door_size[1]/2+0.015, door_size[1]/2-0.015)

        flipped = np.random.binomial(n=1, p=0.5)
        color = (np.random.uniform(0, 1),
                 np.random.uniform(0, 1),
                 np.random.uniform(0, 1))

        color = (1, 0, 0)

        return Door(door_offset, door_size, handle_offset_z, flipped, color, bb_thickness)

class BusyBox(object):
    def __init__(self, width, height, mechanisms, bb_thickness=0.05, file_name=None):
        self._mechanisms = mechanisms
        self._links = []
        self._joints = []
        self._create_skeleton(width, height, bb_thickness)
        self.width = width
        self.height = height
        self.bb_thickness = bb_thickness
        self.file_name = file_name
        self._bb_id = None # set with mechanism ids

    def _create_skeleton(self, width, height, bb_thickness=0.05):
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
                                      urdf.Box(size=(width, bb_thickness, height))
                                  )
                              ),
                              urdf.Visual(
                                  urdf.Origin(xyz=(0, 0, 0), rpy=(0, 0, 0)),
                                  urdf.Geometry(
                                      urdf.Box(size=(width, bb_thickness, height))
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

    def project_onto_backboard(self, pos):
        bb_id = self.get_bb_id()
        p_bb_base_w = p.getLinkState(bb_id,0)[0]
        return [pos[0], p_bb_base_w[1]+self.bb_thickness/2, pos[2]]

    def get_bb_id(self):
        assert self._bb_id is not None, 'BusyBox.set_mechanism_ids() must be called to access pyBullet ids'
        return self._bb_id

    def set_mechanism_ids(self, bb_id):
        self._bb_id = bb_id
        bb_id = self.get_bb_id()
        num_joints = p.getNumJoints(bb_id)
        # joint_id 0 is the busybox back_link
        for joint_id in range(1, num_joints):
            joint_info = p.getJointInfo(bb_id, joint_id)
            link_name = joint_info[12]
            set = False
            for mech in self._mechanisms:
                mech._bb = self
                if mech.handle_name == link_name.decode("utf-8"):
                    mech._handle_id = joint_info[0]
                    set = True
                    break
                if mech.mechanism_type == 'Door' and mech.door_base_name == link_name.decode("utf-8"):
                    mech._door_base_id = joint_info[0]
                    set = True
                    break
        self._bb_id = bb_id

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

    def get_center_pos(self):
        bb_id = self.get_bb_id()
        return p.getLinkState(bb_id,0)[0]

    def set_joint_control_mode(self, mode, maxForce):
        bb_id = self.get_bb_id()
        for jx in range(0, p.getNumJoints(bb_id)):
            p.setJointMotorControl2(bodyUniqueId=bb_id,
                                    jointIndex=jx,
                                    controlMode=mode,
                                    force=maxForce)

    @staticmethod
    def _check_collision(width, height, mechs, mech):
        tree = aabb.AABBTree()
        # Add edge bounding boxes for backboard.
        gripper_gap = 0.035 # so gripper doesn't collide with busybox base
        tree.add(aabb.AABB([(-width/2.0, width/2.0), (height/2.0, height/2.0+1)]))  # top
        tree.add(aabb.AABB([(-width/2.0, width/2.0), (-height/2.0-1, -height/2.0+gripper_gap)]))  # bottom
        tree.add(aabb.AABB([(-width/2.0-1, -width/2.0), (-height/2.0, height/2.0)]))  # left
        tree.add(aabb.AABB([(width/2.0, width/2.0+1), (-height/2.0, height/2.0)]))  # right

        # Get the bounding box for each existing mechanism.
        for ix, m in enumerate(mechs):
            if not m == mech:
                tree.add(m.get_bounding_box(), str(ix))

        # Get the bounding box of the current mechanism and check overlap.
        if tree.does_overlap(mech.get_bounding_box()):
            return True
        else:
            return False

    @staticmethod
    def generate_random_busybox(min_mech=1, max_mech=6, mech_types=[Slider, Door], n_tries=10, urdf_tag='', debug=False):
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
        width, height = 0.6, 0.6
        bb_thickness = 0.05

        # Sample number of mechanisms.
        gen_bb = False
        while not gen_bb:
            mechs = []
            n_mech = np.random.randint(low=min_mech, high=max_mech+1)
            for _ in range(n_mech):
                # For each mechanism pick the type.
                mech_class = np.random.choice(mech_types)
                for _ in range(n_tries):
                    mech = mech_class.random(width, height, bb_thickness)
                    # Check for collisions.
                    if not BusyBox._check_collision(width, height, mechs, mech):
                        mechs.append(mech)
                        break
            try:
                mechs[0]
                gen_bb = True
            except:
                if debug:
                    print('generated a Busybox with no Mechanisms')
                continue

        bb_file = 'models/busybox' + str(urdf_tag) + '.urdf'
        bb = BusyBox(width, height, mechs, bb_thickness, bb_file)

        with open(bb_file, 'w') as handle:
            handle.write(bb.get_urdf())
        return bb

    @staticmethod
    def get_busybox(width, height, mechs, bb_thickness=0.05, urdf_tag=''):
        bb_file = 'models/busybox' + urdf_tag + '.urdf'
        bb = BusyBox(width, height, mechs, bb_thickness, bb_file)
        for mech in mechs:
            if BusyBox._check_collision(width, height, mechs, mech):
                raise Exception('generated a BusyBox with collisions')
        with open(bb_file, 'w') as handle:
            handle.write(bb.get_urdf())

        return bb

    @staticmethod
    def bb_from_result(result, urdf_num=0):
        width, height = 0.6, 0.6
        mech = Mechanism.mech_from_mech_params(result.mechanism_params)
        return BusyBox.get_busybox(width, height, [mech], urdf_tag=str(urdf_num))


def create_simulated_baxter_slider():
    # Create the slider.
    slider = Slider(x_offset=-0.0525,
                    z_offset=-0.19,
                    range=0.33,
                    axis=(1.0, 0),
                    color=(1, 0, 0),
                    bb_thickness=0.05)

    # Create the busybox.
    bb = BusyBox(width=0.6,
                 height=0.6,
                 mechanisms=[slider],
                 bb_thickness=0.05,
                 file_name='models/busybox_real.urdf')
    with open('models/busybox_real.urdf', 'w') as handle:
        handle.write(bb.get_urdf())
    return bb
