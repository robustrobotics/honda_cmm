import pybullet as p
import pybullet_data
import os
import time
import math
import json
import argparse
import numpy as np
from pyquaternion import Quaternion
from collections import defaultdict
from load_UBBDF import loadUBBDF
from gripper import Gripper
from util import transformation, vis_frame
import plotting

FOLDER = '/Users/carismoses/honda_cmm/'
def draw_prismatic(model):
    line_center = np.array([model['rigid_position.x'],
                            model['rigid_position.y'],
                            model['rigid_position.z']])
    axis = np.array([model['prismatic_dir.x'],
                     model['prismatic_dir.y'],
                     model['prismatic_dir.z']])
    lineFrom = (line_center + model['q_min[0]']*axis).tolist()
    lineTo = (line_center + model['q_max[0]']*axis).tolist()

    color = [1, 0, 1]

    p.addUserDebugLine(lineFromXYZ=lineFrom,
                       lineToXYZ=lineTo,
                       lineColorRGB=color,
                       lineWidth=1)
    p.addUserDebugText(text='pri',
                       textPosition=lineTo)

def draw_rigid(model):
    w = 0.025
    h = 0.1  # You can't see the lines if you draw them in the object so draw a bit above.
    line_center = np.array([model['rigid_position.x'],
                            model['rigid_position.y'],
                            model['rigid_position.z']])
    line1From = (line_center + np.array([-w, -w, h])).tolist()
    line1To = (line_center + np.array([w, w, h])).tolist()

    line2From = (line_center + np.array([-w, w, h])).tolist()
    line2To = (line_center + np.array([w, -w, h])).tolist()

    color = [1, 0, 1]
    p.addUserDebugLine(lineFromXYZ=line1From,
                       lineToXYZ=line1To,
                       lineColorRGB=color,
                       lineWidth=1)
    p.addUserDebugLine(lineFromXYZ=line2From,
                       lineToXYZ=line2To,
                       lineColorRGB=color,
                       lineWidth=1)
    p.addUserDebugLine(lineFromXYZ=line_center.tolist(),
                       lineToXYZ=(line_center + np.array([0, 0, h])).tolist(),
                       lineColorRGB=color,
                       lineWidth=1)

    p.addUserDebugText(text='rig',
                       textPosition=line1To)


def draw_revolute(model):
    rot_center = np.array([model['rot_center.x'],
                           model['rot_center.y'],
                           model['rot_center.z']])
    rot_ax = Quaternion(model['rot_axis.w'],
                        model['rot_axis.x'],
                        model['rot_axis.y'],
                        model['rot_axis.z'])
    rot_or = Quaternion(model['rot_orientation.w'],
                        model['rot_orientation.x'],
                        model['rot_orientation.y'],
                        model['rot_orientation.z'])
    rot_axis = (rot_ax * rot_or).get_axis()
    axisFrom = (rot_center + 0.25*rot_axis).tolist()
    axisTo = (rot_center - 0.25*rot_axis).tolist()

    color = [1, 0, 1]

    radius = model['rot_radius']
    if radius < 0.01:
        radius = 0.05
    angles = np.linspace(start=model['q_min[0]'], stop=model['q_max[0]'], num=100)
    for ix in range(1, len(angles)):
        lineFrom = np.array([radius * np.cos(angles[ix]-1),
                             radius * np.sin(angles[ix]-1),
                             0.025])
        lineTo = np.array([radius * np.cos(angles[ix]),
                           radius * np.sin(angles[ix]),
                           0.025])
        p.addUserDebugLine(lineFromXYZ=rot_center+lineFrom,
                           lineToXYZ=rot_center+lineTo,
                           lineColorRGB=color,
                           lineWidth=1)

    p.addUserDebugLine(lineFromXYZ=axisFrom,
                       lineToXYZ=axisTo,
                       lineColorRGB=color,
                       lineWidth=1)
    p.addUserDebugText(text='rot',
                       textPosition=axisFrom)

def draw_joints():
    if not os.path.exists(FOLDER + 'structure.json'):
        return
    with open(FOLDER + 'structure.json', 'r') as handle:
        models = json.load(handle)

    for m in models:
        if m['type'] == 'prismatic':
            draw_prismatic(m)
        elif m['type'] == 'rigid':
            draw_rigid(m)
        elif m['type'] == 'rotational':
            draw_revolute(m)
        else:
            print('cant draw type ' + m['type'] + ' yet')

# TODO: Make forces work with arbitrary revolute/prismatic joints.
def get_force_direction(world, joint_name):
    joint = world['joints'][joint_name]
    joint_position, joint_velocity, _, _ = p.getJointState(bodyUniqueId=world['model_id'],
                                                           jointIndex=joint.pybullet_id)
    lower, upper = joint.lower, joint.upper
    if joint_velocity > 0:
        direction = 1
        if not upper and math.isclose(upper, joint_position, abs_tol=0.01):
            direction = -1
    else:
        direction = -1
        if not lower is None and math.isclose(lower, joint_position, abs_tol=0.01):
            direction = 1

    return direction


def actuate_revolute(world, gripper, joint_name):
    # TODO: need to make a function that uses the joint info to return a 3D direction
    # (unit vector) in world frame of desired motion of the joint handle
    #direction = get_force_direction(world, joint_name)
    delta_theta = -.5
    p_handle_w = p.getLinkState(world['model_id'], world['links']['spinner_handle'].pybullet_id)[0]
    p_base_w, orn_base_w = p.getLinkState(world['model_id'], world['links']['spinner'].pybullet_id)[:2]
    p_handle_base = np.subtract(p_handle_w, p_base_w)
    R = np.linalg.norm(p_handle_base[:2])
    if p_handle_base[1] > 0:
        theta = np.arccos(p_handle_base[0]/R)
    else:
        theta = -np.arccos(p_handle_base[0]/R)
    theta_new = theta + delta_theta
    p_handle_w_des = np.add([p_base_w[0], p_base_w[1], 0], [R*np.cos(theta_new), R*np.sin(theta_new), p_handle_w[2]])
    direction = np.subtract(p_handle_w_des, p_handle_w)
    unit_vector = np.divide(direction, np.linalg.norm(direction))
    magnitude = .1
    gripper.apply_force(np.multiply(magnitude, unit_vector))

def actuate_prismatic(world, gripper, joint_name):
    # TODO: need to make a function that uses the joint info to return a 3D direction
    # (unit vector) in world frame of desired motion of the joint handle
    #direction = get_force_direction(world, joint_name)
    unit_vector = [0., -1., 0.]
    magnitude = .1
    gripper.apply_force(np.multiply(magnitude, unit_vector))

def log_poses(world, joint_name, log_name, log):
    if joint_name == 'all':
        to_log = world['links'].keys()
    else:
        to_log = [world['joints'][joint_name].child_link, 'base_link']


    for link_name in to_log:
        if link_name == 'base_link':
            position, orientation = p.getBasePositionAndOrientation(bodyUniqueId=world['model_id'])
        else:
            position, orientation = p.getLinkState(bodyUniqueId=world['model_id'],
                                                   linkIndex=world['links'][link_name].pybullet_id)[0:2]
            if link_name == 'spinner_handle':
                orientation = p.getLinkState(world['model_id'], world['links']['spinner'].pybullet_id)[1]
        log[link_name].append(position + orientation)

    if 'spinner' in log:
        del log['spinner']

    with open(FOLDER + log_name, 'w') as handle:
        json.dump(log, handle)



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--joint-name', default='all', type=str, help='[<link_name>|all]')
    parser.add_argument('--log-name', default='', type=str, help='JSON file name')
    parser.add_argument('--duration', default=3, type=int)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    # Set PyBullet configuration.
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(0)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=30,
        cameraPitch=-52,
        cameraTargetPosition=(0., 0., 0.))
    timestep = 1. / 100.

    # Load models.
    plane_id = p.loadURDF("plane.urdf")
    world = loadUBBDF(urdf_file='models/{0}/model.urdf'.format(args.model_name),
                      ubbdf_file='models/{0}/model_relations.ubbdf'.format(args.model_name))
    gripper = Gripper()

    # The joint motors need to be disabled before we can apply forces to them.
    maxForce = 0
    mode = p.VELOCITY_CONTROL
    for ix in range(0, p.getNumJoints(world['model_id'])):
        p.setJointMotorControl2(bodyUniqueId=world['model_id'],
                                jointIndex=ix,
                                controlMode=mode,
                                force=maxForce)

    # Simulation loop.
    log = defaultdict(list)
    if args.visualize:
        draw_joints()

    # actuate each joint sequentially with the gripper
    gripper_orn = p.getQuaternionFromEuler([np.pi, 0., 0.])
    #for joint_name, joint in world['joints'].items():
        #if joint_name == args.joint_name or args.joint_name == 'all':
    for joint_name in ['spinner_handle', 'slider']:
        joint = world['joints'][joint_name]
        link_name = joint.child_link

        # grasp the joint handle
        p_link = p.getLinkState(world['model_id'], world['links'][link_name].pybullet_id)[0]
        if joint_name == 'spinner_handle':
            p_link = np.add(p_link, [0., 0., .02])
        gripper.apply_force(finger_state='open')
        gripper.set_tip_pose([p_link, gripper_orn])
        gripper.apply_force(finger_state='close')

        # actuate prismatic joints
        if joint.type == 'prismatic':
            for tx in range(0, args.duration*10):
                actuate_prismatic(world, gripper, joint_name)
                for r in world['relations']:
                    r.update(world=world,
                             parent_joint=world['joints'][r.parent_joint],
                             child_joint=world['joints'][r.child_joint],
                             params=r.params)

                # Log poses.
                if tx % 5 == 0 and len(args.log_name) > 0:
                    log_poses(world, args.joint_name, args.log_name, log)

        # actuate revolute joints
        elif joint.type == 'revolute' or joint.type == 'continuous':
            for tx in range(0, args.duration*10):
                actuate_revolute(world, gripper, joint_name)
                for r in world['relations']:
                    r.update(world=world,
                             parent_joint=world['joints'][r.parent_joint],
                             child_joint=world['joints'][r.child_joint],
                             params=r.params)

                # Log poses.
                if tx % 5 == 0 and len(args.log_name) > 0:
                    log_poses(world, args.joint_name, args.log_name, log)

    p.disconnect()
