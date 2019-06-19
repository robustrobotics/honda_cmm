from util import util
import numpy as np
import argparse
import pybullet as p
import pybullet_data
from actions.gripper import Gripper
from gen.generator_busybox import BusyBox
from actions import policies
from collections import namedtuple

def test_policy(viz=False, debug=False, max_mech=6, random=False, k=None, d=None,\
                    add_dist=None, p_err_thresh=None, p_delta=None, git_hash=None):
    if not viz:
        client = p.connect(p.DIRECT)
    else:
        client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)

    bb = BusyBox.generate_random_busybox(max_mech=max_mech)
    try:
        mech = bb._mechanisms[0]
    except:
        if debug:
            print('generated a Busybox with no Mechanisms')
        p.disconnect(client)
        return []
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=180,
        cameraPitch=0,
        cameraTargetPosition=(0., 0., bb.height/2))

    plane_id = p.loadURDF("plane.urdf")

    #p.setGravity(0, 0, -10)

    bb_file = 'models/busybox.urdf'
    with open(bb_file, 'w') as handle:
        handle.write(bb.get_urdf())
    model = p.loadURDF(bb_file, [0., -.3, 0.])
    bb.set_mechanism_ids(model)
    maxForce = 10
    mode = p.VELOCITY_CONTROL
    for jx in range(0, p.getNumJoints(model)):
        p.setJointMotorControl2(bodyUniqueId=model,
                                jointIndex=jx,
                                controlMode=mode,
                                force=maxForce)

    # can change resolution and shadows with this call
    image_data_pybullet = p.getCameraImage(205, 154, shadow=0) # do before add gripper to world
    image_data = util.ImageData(*image_data_pybullet[:3])

    gripper = Gripper(model, k, d, add_dist, p_err_thresh)

    results = []
    for mech in bb._mechanisms:
        # parameters for grasping
        p_joint_world_init = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        q_tip_world_init = np.array([0.50019904,  0.50019904, -0.49980088, 0.49980088])

        # try correct policy on each
        if not random:
            if mech.mechanism_type == 'Door':
                policy = policies.Revolute.model(bb, mech, p_delta)
                config_goal = -np.pi/2 if mech.flipped else np.pi/2
            elif mech.mechanism_type == 'Slider':
                policy = policies.Prismatic.model(bb, mech, p_delta)
                config_goal = .1
        # else generate a random policy and goal config
        else:
            policy = policies.generate_random_policy(bb, p_delta)
            config_goal = policy.generate_random_config()
        p_joint_base_world_init = bb.project_onto_backboard(p_joint_world_init)
        p_tip_world_init = np.add(p_joint_world_init, [0., .015, 0.])
        pose_tip_world_init = util.Pose(p_tip_world_init, q_tip_world_init)
        traj = policy.generate_trajectory(pose_tip_world_init, p_joint_base_world_init, config_goal, debug)

        # execute trajectory
        waypoints_reached, duration, joint_motion, pose_joint_world_final = \
                gripper.execute_trajectory(pose_tip_world_init, traj, mech, debug=debug)

        # save result data
        control_params = util.ControlParams(gripper.k, gripper.d, gripper.add_dist, gripper.p_err_thresh, policy.p_delta)
        policy_params = policy.get_params_tuple()
        mechanism_params = mech.get_params_tuple()
        results += [util.Result(control_params, policy_params, mechanism_params, waypoints_reached,\
                    joint_motion, pose_joint_world_final, image_data, git_hash)]

    p.disconnect(client)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max-mech', type=int, default=6)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    test_policy(args.viz, args.debug, args.max_mech, args.random)
    print('done testing policy')
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting...')
