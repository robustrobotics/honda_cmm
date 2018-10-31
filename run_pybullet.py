import pybullet as p
import pybullet_data
import time
from loadUBBDF import loadUBBDF

def turnOnLight(model):
    p.changeVisualShape(objectUniqueId=model,
                        linkIndex=3,
                        rgbaColor=[1, 1, 0, 1])


def spinSpinner(model):
    p.applyExternalForce(objectUniqueId=model,
                         linkIndex=2,
                         forceObj=[0., -1, 0],
                         posObj=[0, 0, 0],
                         flags=p.LINK_FRAME)


def slideSlider(model, force):
    print(p.getJointState(model, 0))
    p.applyExternalForce(objectUniqueId=model,
                         linkIndex=0,
                         forceObj=[0., force, 0],
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
    timestep = 1./255.

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
            slideSlider(world['model_id'], -1)
        elif timestep*tx > 1.0:
            slideSlider(world['model_id'], 1)
        else:
            spinSpinner(world['model_id'])

        for r in world['relations']:
            r.update(world=world,
                     parent_joint=world['joints'][r.parent_joint],
                     child_joint=world['joints'][r.child_joint],
                     params=r.params)

        p.stepSimulation()
        time.sleep(timestep)

    p.disconnect()
