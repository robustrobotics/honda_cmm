from util import util
import numpy as np
import argparse
import pybullet as p
import pybullet_data
from actions.gripper import Gripper
from gen.generator import BusyBox
from actions.policies import generate_random_policy, Prismatic, Revolute
from collections import namedtuple

Result = namedtuple('Result', 'gripper policy mechanism waypoints_reached duration motion final_pose image')
"""
Result contains the performance information after the gripper tries to move a mechanism

:param gripper: gripper.Gripper object
:param policy: policies.Policy object
:param mechanism: data.Mechanism object
:param waypoints_reached: percentage of waypoints reached when attempting to move mechanism
:param duration: execution duration before success or timeout
:param motion: the net distance the mechanism handle moved
:param final_pose: the final_pose of the gripper tip if it is in contact with the mechanism
                    at completion, else None
:param image: type returned by p.getCameraImage ([2:4] are RGB and depth values)
"""

def test_policy(viz=False, debug=False, max_mech=6, random=False, k=None, d=None,\
                    add_dist=None, p_err_eps=None, delta_pos=None):
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
            print('generated a Busyox with no Mechanisms')
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
    image = p.getCameraImage(1024,768) # do before add gripper to world
    gripper = Gripper(model, k, d, add_dist, p_err_eps)

    results = []
    for mech in bb._mechanisms:
        # parameters for grasping
        p_mech_w = p.getLinkState(bb.bb_id, mech.handle_id)[0]
        q_t_w_des = np.array([0.50019904,  0.50019904, -0.49980088, 0.49980088])

        # try correct policy on each
        if not random:
            if mech.mechanism_type == 'Door':
                policy = Revolute.model(bb, mech, delta_pos)
                goal_q = -np.pi/2 if mech.flipped else np.pi/2
            elif mech.mechanism_type == 'Slider':
                policy = Prismatic.model(bb, mech, delta_pos)
                goal_q = .1
        # else generate a radom policy and goal config
        else:
            policy = generate_random_policy(bb, delta_pos)
            goal_q = policy.generate_random_config()
        init_joint_pos = bb.project_onto_backboard(p_mech_w)
        grasp_pos = np.add(p_mech_w, [0., .015, 0.])
        grasp_pose = util.Pose(grasp_pos, q_t_w_des)
        traj = policy.generate_trajectory(grasp_pose, init_joint_pos, goal_q, debug)

        # execute trajectory
        waypoints_reached, duration, motion = gripper.execute_trajectory(grasp_pose, traj, mech, debug=debug)
        if gripper.in_contact(mech):
            final_pose = p.getLinkState(bb.bb_id, mech.handle_id)
        else:
            final_pose = None
        results += [Result(gripper, policy, mech, waypoints_reached, duration,\
                    motion, final_pose, image)]

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

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting...')
