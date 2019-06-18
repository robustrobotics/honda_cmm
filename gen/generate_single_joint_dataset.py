import argparse
import numpy as np
import os
import pybullet as p
import pybullet_data
from actions.gripper import Gripper
from util import util
from gen.generator_busybox import BusyBox, Door, Slider
import cv2
import pickle

class JointLogger(object):
    def __init__(self, folder):
        self.datasets = []
        self.ix = 0
        self.dataset = []
        self.folder = folder
        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.labels = []
        self.imgs = []

    def start_dataset(self):
        self.dataset = []

    def take_photo(self):
        #cam_info = p.getDebugVisualizerCamera()
        #print(cam_info[2], cam_info[3])
        view_mat = (-0.8660253882408142, 0.43301281332969666, -0.2500000298023224, 0.0, -0.5000001192092896, -0.75, 0.43301263451576233, 0.0, 0.0, 0.5, 0.866025447845459, 0.0, 0.0, 0, -0.6999999284744263, 1.0)
        proj_mat = (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)

        h, w, rgb, depth, seg = p.getCameraImage(1000, 1000, view_mat, proj_mat)
        img = np.array(rgb, dtype='int').reshape(1000, 1000, 4)
        # PyBullet has RGB but opencv user BGR.
        tmp_red = img[:, :, 0].tolist()
        img[:, :, 0] = img[:, :, 2]
        img[:, :, 2] = np.array(tmp_red)
        img = img[:, :, :3]

        cv2.imwrite('{0}/busybox_{1}.png'.format(self.folder, ix), img)
        return img

    def end_dataset(self, bb):
        if len(self.dataset) > 1:
            self.datasets.append(self.dataset)

            label = bb._mechanisms[0].mechanism_type
            if label == 'Door':
                label = '{0}_{1}'.format(label, int(bb._mechanisms[0].flipped))
            self.labels.append(label)
            # TODO: Save URDF to file.
            with open('{0}/busybox_{1}.urdf'.format(self.folder, self.ix), 'w') as handle:
                handle.write(bb.get_urdf())
            # TODO: Save image showing motion in URDF.
            #self.take_photo()
            #final_img = self.imgs[0]
            #for i in self.imgs[1:]:
            #    final_img = final_img + i
            #final_img = final_img / len(self.imgs)
            #cv2.imwrite('{0}/busybox_{1}.png'.format(self.folder, 'final'), final_img)
            self.ix += 1

    def log_state(self, bb):
        # TODO: Log the busy box URDF path that generated this trajectory.
        # TODO: Get pose of base and handle as well as transform between the two.
        handle_id = bb._mechanisms[0].handle_id
        pos, orn, _, _, _, _ = p.getLinkState(bodyUniqueId=bb.bb_id,
                                              linkIndex=handle_id)
        to_pos = (pos[0]+0.01, pos[1]+0.01, pos[2]+0.01)
        p.addUserDebugLine(lineFromXYZ=pos,
                           lineToXYZ=to_pos,
                           lineColorRGB=(1, 0, 0))

        self.dataset.append(pos+orn)
        #self.imgs.append(self.take_photo())

    def save(self):
        with open('{0}/dataset_test.pkl'.format(self.folder), 'wb') as handle:
            pickle.dump((self.datasets, self.labels), handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    logger = JointLogger(args.folder)

    for ix in range(args.n):
        logger.start_dataset()

        if not args.viz:
            client = p.connect(p.DIRECT)
        else:
            client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.7,
            cameraYaw=210,
            cameraPitch=-60,
            cameraTargetPosition=(0., 0., 0.))
        plane_id = p.loadURDF("plane.urdf")

        print('Busybox Number:', ix)
        bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider])

        p.setGravity(0, 0, -10)
        bb_file = 'models/busybox_{0}.urdf'.format(ix)
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

        logger.end_dataset(bb)

        p.disconnect(client)

    logger.save()

    for d in logger.datasets:
        print(len(d))
    print(logger.datasets[0][0:100:30])
