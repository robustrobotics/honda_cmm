import pybullet as p
import pybullet_data
import time
from loadUBBDF import loadUBBDF


def spinSpinner(world):
    p.applyExternalForce(objectUniqueId=world['model_id'],
                         linkIndex=world['links']['spinner_handle'].pybullet_id,
                         forceObj=[0., -1, 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME)


def slideSlider(world, force):
    p.applyExternalForce(objectUniqueId=world['model_id'],
                         linkIndex=world['links']['slider'].pybullet_id,
                         forceObj=[0., force, 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME)


def turnHandle(world, force):
    p.applyExternalForce(objectUniqueId=world['model_id'],
                         linkIndex=world['links']['door_handle'].pybullet_id,
                         forceObj=[force, 0., 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME)


if __name__ == '__main__':
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
    timestep = 1./100.

    # Load models.
    plane_id = p.loadURDF("plane.urdf")

    world = loadUBBDF(urdf_file='models/busybox/model.urdf',
                      ubbdf_file='models/busybox/model_relations.ubbdf')

    # The joint motors need to be disabled before we can apply forces to them.
    maxForce = 0
    mode = p.VELOCITY_CONTROL
    for ix in range(0, p.getNumJoints(world['model_id'])):
        p.setJointMotorControl2(bodyUniqueId=world['model_id'],
                                jointIndex=ix,
                                controlMode=mode,
                                force=maxForce)

    # Simulation loop.
    for tx in range(0, 30000):
        if timestep*tx > 2.0:
            slideSlider(world, -1)
            turnHandle(world, -1)
        elif timestep*tx > 1.0:
            slideSlider(world, 1)
            turnHandle(world, 1)
        else:
            spinSpinner(world)

        for r in world['relations']:
            r.update(world=world,
                     parent_joint=world['joints'][r.parent_joint],
                     child_joint=world['joints'][r.child_joint],
                     params=r.params)

        p.stepSimulation()
        time.sleep(timestep)

    p.disconnect()
