import numpy as np
import odio_urdf as urdf
import argparse
import pybullet as p
import pybullet_data
import aabbtree as aabb
from actions.gripper import Gripper
from util import util
from gen.generator_busybox import BusyBox


class JointLogger(object):
    def __init__(self):
        self.datasets = []
        self.ix = 0

    def start_dataset(self):
        self.dataset = []

    def end_dataset(self):
        self.datasets.append(self.dataset)
        self.ix += 1

    def log_state(self, bb):
        # TODO: Log the busy box URDF path that generated this trajectory.
        # TODO: Get pose of base and handle as well as transform between the two.
        handle_id = bb._mechanisms[0].handle_id
        pos, orn, _, _, _, _ = p.getLinkState(bodyUniqueId=bb.bb_id,
                                              linkIndex=handle_id)
        self.dataset.append(pos+orn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    logger = JointLogger()

    for ix in range(args.n):
        logger.start_dataset()

        if not args.viz:
            client = p.connect(p.DIRECT)
        else:
            client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=180,
            cameraPitch=-30,
            cameraTargetPosition=(0., 0., 0.))
        plane_id = p.loadURDF("plane.urdf")

        print('Busybox Number:',ix)
        bb = BusyBox.generate_random_busybox(max_mech=1)

        bb_file = 'models/busybox_{0}.urdf'.format(ix)
        if args.save:
            with open(bb_file, 'w') as handle:
                handle.write(bb.get_urdf())

        p.setGravity(0, 0, -10)

        with open(bb_file, 'w') as handle:
            handle.write(bb.get_urdf())
        model = p.loadURDF(bb_file, [0., -.3, 0.])
        bb.set_mechanism_ids(model)
        bb.set_joint_models(model)
        maxForce = 10
        mode = p.VELOCITY_CONTROL
        for jx in range(0, p.getNumJoints(model)):
            p.setJointMotorControl2(bodyUniqueId=model,
                                    jointIndex=jx,
                                    controlMode=mode,
                                    force=maxForce)

        gripper = Gripper(model, 'traj')
        bb.actuate_joints(model, gripper, 'traj', args.debug, args.viz, callback=logger.log_state)
        print('done actuating')

        p.disconnect(client)

        logger.end_dataset()

    for d in logger.datasets:
        print(len(d))
    print(logger.datasets[0][0:100:30])
