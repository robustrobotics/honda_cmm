import util
import numpy as np
import argparse
import pybullet as p
import pybullet_data
from gripper import Gripper
from joints import Prismatic, Revolute
from data.generator import BusyBox, Slider, Door
from policies import Rev, Prism, generate_random_policy

parser = argparse.ArgumentParser()
parser.add_argument('--viz', action='store_true')
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--max-mech', type=int, default=6)
parser.add_argument('--random', action='store_true')
args = parser.parse_args()

if args.debug:
    import pdb; pdb.set_trace()

if not args.viz:
    client = p.connect(p.DIRECT)
else:
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING,0)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)

bb = BusyBox.generate_random_busybox(max_mech=args.max_mech)

camera_center = (0., 0., bb.height/2)
p.resetDebugVisualizerCamera(
    cameraDistance=0.2,
    cameraYaw=180,
    cameraPitch=0,
    cameraTargetPosition=camera_center)
plane_id = p.loadURDF("plane.urdf")

#p.setGravity(0, 0, -10)

bb_file = 'models/busybox.urdf'
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

gripper = Gripper(model)
for mech in bb._mechanisms:
    # Good grasping parameters for testing
    p_mech_w = p.getLinkState(bb.bb_id, mech.handle_id)[0]
    q_t_w_des = np.array([0.50019904,  0.50019904, -0.49980088, 0.49980088])

    # try correct policy on each
    if not args.random:
        if mech.mechanism_type == 'Door':
            goal_q = -np.pi/2 if mech.flipped else np.pi/2
            policy = Rev(mech.joint_model.rot_center,
                            mech.joint_model.rot_axis,
                            mech.joint_model.rot_radius,
                            mech.joint_model.rot_orientation,
                            goal_q)
        elif mech.mechanism_type == 'Slider':
            policy = Prism(mech.joint_model.rigid_position,
                            mech.joint_model.rigid_orientation,
                            mech.joint_model.prismatic_dir,
                            .1)
    else:
        policy = generate_random_policy(bb)
    init_joint_pos = bb.project_onto_backboard(p_mech_w)
    grasp_pos = np.add(p_mech_w, [0., .015, 0.])
    grasp_pose = util.Pose(grasp_pos, q_t_w_des)
    traj = policy.generate_trajectory(grasp_pose, init_joint_pos, args.debug)

    # execute trajectory
    gripper.execute_trajectory(grasp_pose, traj, debug=args.debug)

p.disconnect(client)

try:
    while True:
        pass
except KeyboardInterrupt:
    print('Exiting...')
