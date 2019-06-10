import util
import numpy as np
import argparse
import pybullet as p
import pybullet_data
from gripper import Gripper
from joints import Prismatic, Revolute
from data.generator import BusyBox
import policies

parser = argparse.ArgumentParser()
parser.add_argument('--viz', action='store_true')
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--max-mech', type=int, default=6)
args = parser.parse_args()

if args.debug:
    import pdb; pdb.set_trace()

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

control_method = 'traj'
gripper = Gripper(model, control_method)
for mech in bb._mechanisms:
    # Good grasping parameters for testing
    p_t_w = p.getLinkState(bb.bb_id, mech.handle_id)[0]
    q_t_w =  [0.50019904,  0.50019904, -0.49980088, 0.49980088]
    pose_t_w_des = util.Pose(p_t_w, q_t_w)

    # try correct policy on each
    if mech.mechanism_type == 'Door':
        goal_q = -np.pi/2 if mech.flipped else np.pi/2
        params = policies.RevParams(pose_t_w_des,
                            mech.joint_model.rot_center,
                            mech.joint_model.rot_axis,
                            mech.joint_model.rot_radius,
                            mech.joint_model.rot_orientation,
                            goal_q)
        traj = policies.rev(params)
    elif mech.mechanism_type == 'Slider':
        params = policies.PrismParams(pose_t_w_des,
                                mech.joint_model.rigid_position,
                                mech.joint_model.rigid_orientation,
                                mech.joint_model.prismatic_dir,
                                .1)
        traj = policies.prism(params)

    # testing PD control
    p_t_w_init = [0., 0., .2]
    q_t_w_init = [0.50019904,  0.50019904, -0.49980088, 0.49980088]
    pose_t_w_init = util.Pose(p_t_w_init, q_t_w_init)
    for t in range(10):
        gripper.set_tip_pose(pose_t_w_init, reset=True)

    traj_x = np.arange(0., 1., .01)
    traj = [util.Pose([x, 0., .2], q_t_w_init) for x in traj_x]
    command = util.Command(traj=traj)
    gripper.apply_command(command, viz=args.viz)

    '''
    # execute traj
    command = util.Command(traj=traj)
    gripper.grasp_handle(traj[0], args.viz)
    gripper.apply_command(command, viz=args.viz)
    '''

p.disconnect(client)

try:
    while True:
        pass
except KeyboardInterrupt:
    print('Exiting...')
