from run_pybullet import setupWorld
import pybullet as p
import pybullet_data
import random
from random import randint, gauss
import time

def make_rand_vector():
    vec = [gauss(0, 1) for i in range(3)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def random_act(world):
    timestep = 1./100.

    # select random link (except for base_link) and force direction
    link_name = 'base_link'
    while link_name == 'base_link':
        link_name = random.choice(world['links'].keys())
    force_mag = 1.
    force_dir = make_rand_vector()

    # apply action
    duration = 10
    for t in range(duration):
        p.applyExternalForce(objectUniqueId=world['model_id'],
                             linkIndex=world['links'][link_name].pybullet_id,
                             forceObj=force_dir,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME)
        p.stepSimulation()
        time.sleep(timestep)

    return link_name

def runEstimator(world, steps):


    for i in range(steps):
        # take an action at random
        link_name = random_act(world)

        # update model


if __name__ == '__main__':
    import pdb; pdb.set_trace()
    # Set PyBullet configuration.
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    world = setupWorld()
    runEstimator(world, 50)

    p.disconnect()
