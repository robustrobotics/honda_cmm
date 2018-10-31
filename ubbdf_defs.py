# classes used for storing information from the parsed UBBDF
import pybullet as p


class Link(object):
    def __init__(self, name):
        self.name = name


class Joint(object):
    def __init__(self, name, type, child_link):
        self.name = name
        self.type = type
        self.child_link = child_link


class Relation(object):
    def __init__(self, name, parent_joint, child_joint, type, params):
        self.name = name
        self.parent_joint = parent_joint
        self.child_joint = child_joint
        self.type = type    # type in {light, door, sound}
        self.params = params

        # bind update functions here based on given type
        if self.type == 'light':
            self.update = update_light
        elif self.type == 'sound':
            self.update = update_sound
        elif self.type == 'door':
            self.update = update_door

# relation functions for checking if relation has been activated and updating the
# world if it has


# discrete examples
def update_light(world, parent_joint, child_joint, params):
    """
    :param world: A dictionary describing the busybox model (links, joints, relations).
    :param parent_joint: The name of the joint describing the activation constraint.
    :param child_joint: The name of the joint whose color will change upon activation.
    :param params: The following params should be specified in the xml:
                    threshold: the value of the parent_joint to toggle the light
                    off_color: color of the deactivated light
                    on_color: color of the activated light
    """
    on_color = [float(s) for s in params['on_color'].split()]
    off_color = [float(s) for s in params['off_color'].split()]
    joint_state = p.getJointState(world['model_id'], parent_joint.pybullet_id)[0]

    if joint_state > float(params['threshold']):
        p.changeVisualShape(objectUniqueId=world['model_id'],
                            linkIndex=world['links'][child_joint.child_link].pybullet_id,
                            rgbaColor=on_color)
    else:
        p.changeVisualShape(objectUniqueId=world['model_id'],
                            linkIndex=world['links'][child_joint.child_link].pybullet_id,
                            rgbaColor=off_color)


def update_sound(world, parent_joint, child_joint, params):
    return True


# continuous example
def update_door(world, parent_joint, child_joint, params):
    return False
