import pybullet as p
import pybullet_data
import time
import math
import json
import argparse
from collections import defaultdict
from load_UBBDF import loadUBBDF


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


def actuate_revolute(world, joint_name):
    joint = world['joints'][joint_name]
    link_name = joint.child_link

    direction = get_force_direction(world, joint_name)

    p.applyExternalTorque(objectUniqueId=world['model_id'],
                          linkIndex=world['links'][link_name].pybullet_id,
                          torqueObj=[0, 0, direction*0.01],
                          flags=p.WORLD_FRAME)


def actuate_prismatic(world, joint_name):
    joint = world['joints'][joint_name]
    link_name = joint.child_link

    direction = get_force_direction(world, joint_name)

    p.applyExternalForce(objectUniqueId=world['model_id'],
                         linkIndex=world['links'][link_name].pybullet_id,
                         forceObj=[0., direction*0.5, 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME)


def log_poses(world, joint_name, log_type, log_name, log):
    if joint_name == 'all':
        to_log = world['links'].keys()
    else:
        to_log = [world['joints'][joint_name].child_link, 'base_link']

    if log_type == 'json':

        for link_name in to_log:
            if link_name == 'base_link':
                position, orientation = p.getBasePositionAndOrientation(bodyUniqueId=world['model_id'])
            else:
                position, orientation = p.getLinkState(bodyUniqueId=world['model_id'],
                                                       linkIndex=world['links'][link_name].pybullet_id)[0:2]
            log[link_name].append(position + orientation)

        with open(log_name, 'w') as handle:
            json.dump(log, handle)

    elif log_type == 'ros':
        print('ROS logging not implemented.')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True, type=str)
    parser.add_argument('--joint-name', required=True, type=str, help='[<link_name>|all]')
    parser.add_argument('--log-type', required=True, type=str, help='[json|ros]')
    parser.add_argument('--log-name', required=True, type=str, help='ROS topic name or JSON file name')
    parser.add_argument('--duration', default=3, type=int)
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
    for tx in range(0, args.duration*100):
        # Actuate the specified joints.
        for joint_name, joint in world['joints'].items():
            if joint_name == args.joint_name or args.joint_name == 'all':
                if joint.type == 'revolute' or joint.type == 'continuous':
                    actuate_revolute(world, joint_name)
                elif joint.type == 'prismatic':
                    actuate_prismatic(world, joint_name)

        # Update causal relations.
        for r in world['relations']:
            r.update(world=world,
                     parent_joint=world['joints'][r.parent_joint],
                     child_joint=world['joints'][r.child_joint],
                     params=r.params)

        # Log poses.
        log_poses(world, args.joint_name, args.log_type, args.log_name, log)

        p.stepSimulation()
        time.sleep(timestep)

    p.disconnect()
