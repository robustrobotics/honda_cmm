import pybullet as p
import pybullet_data
from util import util

def setup_env(bb, viz=False, debug=False):
    if not viz:
        client = p.connect(p.DIRECT)
    else:
        client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)

    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=180,
        cameraPitch=0,
        cameraTargetPosition=(0., 0., bb.height/2))

    plane_id = p.loadURDF("plane.urdf")
    model = p.loadURDF(bb.file_name, [0., -.3, 0.])
    bb.set_mechanism_ids(model)

    #p.setGravity(0, 0, -10)
    maxForce = 10
    mode = p.VELOCITY_CONTROL
    for jx in range(0, p.getNumJoints(bb.bb_id)):
        p.setJointMotorControl2(bodyUniqueId=bb.bb_id,
                                jointIndex=jx,
                                controlMode=mode,
                                force=maxForce)

    # can change resolution and shadows with this call
    view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=0.4,
                                                      yaw=180,
                                                      pitch=0,
                                                      roll=0,
                                                      upAxisIndex=2,
                                                      cameraTargetPosition=(0., 0., bb.height / 2))

    aspect = 205. / 154.
    nearPlane = 0.01
    farPlane = 100
    fov = 60
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

    image_data_pybullet = p.getCameraImage(205, 154, shadow=0, renderer=p.ER_TINY_RENDERER, viewMatrix=view_matrix, projectionMatrix=projection_matrix)  # do before add gripper to world
    image_data = util.ImageData(*image_data_pybullet[:3])

    p.stepSimulation()
    return image_data

def custom_bb():
    """ Generate a custom BusyBox environment
    """
    bb_width = 0.8
    bb_height = 0.4
    door_offset = (.075, -0.09)
    door_size = (0.15, 0.15)
    handle_offset = -0.15/2 +.015
    flipped = True
    door = Door(door_offset, door_size, handle_offset, flipped, color=[1,0,0])
    return BusyBox.get_busybox(bb_width, bb_height, [door])
