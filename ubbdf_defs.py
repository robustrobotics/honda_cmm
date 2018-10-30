# classes used for storing information from the parsed UBBDF
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
    return True

def update_sound(world, parent_joint, child_joint, params):
    return True

# continuous example
def update_door(world, parent_joint, child_joint, params):
    return False
