import pybullet as p
import pybullet_data
from gen.generator_busybox import BusyBox, Slider, Door
from actions.gripper import Gripper
from util import util

def random_env(viz=False, k=None, d=None, add_dist=None, p_err_thresh=None, max_mech=1,
                mech_types = [Door, Slider], debug=False, urdf_tag=''):

    if not viz:
        client = p.connect(p.DIRECT)
    else:
        client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)

    # this generates the BusyBox object and loads the URDF
    bb = BusyBox.generate_random_busybox(max_mech=max_mech, mech_types=[Door, Slider], urdf_tag=urdf_tag)

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

    gripper = Gripper(bb.bb_id, k, d, add_dist, p_err_thresh)
    return bb, gripper, image_data
