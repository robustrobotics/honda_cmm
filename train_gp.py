from data.generator import BusyBox
import pybullet as p
import pybullet_data
import argparse
from collections import namedtuple
from gripper import Gripper
from gp_learner import GPLearner

VisualFeatures = namedtuple('VisualFeatures', 'color')# base_pose handle_pose')
JointParameters = namedtuple('JointParameters', 'joint_type')# 'joint_parameters')
PrismaticParameters = namedtuple('PrismaticParameters', 'axis_pose direction')
RevoluteParameters = namedtuple('RevoluteParameters', 'rot_center_pose radius handle_pose')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--actuate', action='store_true')
    parser.add_argument('--control-method', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-mech', type=int, default=6)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    datasets = []
    for ix in range(args.n):
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

        bb = BusyBox.generate_random_busybox(max_mech=args.max_mech)

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

        # make dataset of handle_name : [VisualFeatures]
        dataset = {}
        for mech in bb._mechanisms:
            dataset[mech.handle_name] = [VisualFeatures(mech.color)]

        if args.actuate:
            gripper = Gripper(model, args.control_method)
            bb.actuate_joints(model, gripper, args.control_method, args.debug, args.viz)#, bb_learner=bb_learner)

        # add joint params to dataset (need to add name of joint to dataset (find in msg))
        for mech in bb._mechanisms:
            if mech.mechanism_type == 'Slider':
                joint_type = 'prismatic'
            elif mech.mechanism_type == 'Door':
                joint_type = 'revolute'
            dataset[mech.handle_name] += [JointParameters(joint_type)]
        datasets += [dataset]

        p.disconnect(client)

    gp_learner = GPLearner()
    gp_learner.get_model(datasets)
