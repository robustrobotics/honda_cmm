import pybullet as p
import xml.etree.ElementTree as ET
from ubbdf_defs import Link, Joint, Relation


def loadUBBDF(urdf_file, ubbdf_file):
    # get links and joints from urdf
    urdf_tree = ET.parse(urdf_file)
    urdf_root = urdf_tree.getroot()

    links = {}
    for link in urdf_root.findall('link'):
        name = link.attrib['name']
        links[name] = Link(name)

    joints = {}
    for joint in urdf_root.findall('joint'):
        name = joint.attrib['name']
        type = joint.attrib['type']
        for j_elem in joint:
            if j_elem.tag == 'child':
                child_link = j_elem.attrib['link']

        limits = joint.findall('limit')

        if len(limits) != 0 and 'lower' in limits[0].keys():
            lower = float(limits[0].attrib['lower'])
            upper = float(limits[0].attrib['upper'])
            joints[name] = Joint(name, type, child_link, lower, upper)
        else:
            joints[name] = Joint(name, type, child_link)


    # get relations from ubbdf
    ubbdf_tree = ET.parse(ubbdf_file)
    ubbdf_root = ubbdf_tree.getroot()

    relations = []
    for relation in ubbdf_root:
        r_att = relation.attrib
        name = r_att['name']
        type = r_att['type']
        for r_elem in relation:
            if r_elem.tag == 'parent_joint':
                parent_joint = r_elem.attrib['name']
            if r_elem.tag == 'child_joint':
                child_joint = r_elem.attrib['name']
            if r_elem.tag == 'parameters':
                params = r_elem.attrib
        relations += [Relation(name, parent_joint, child_joint, type, params)]

    # associate pybullet link and joint ids with each link/joint
    model = p.loadURDF(urdf_file)
    print('-----')
    for ix in range(p.getNumJoints(model)):
        info = p.getJointInfo(model, ix)
        joint_name, link_name = info[1].decode('utf-8'), info[12].decode('utf-8')
        print(joint_name, link_name)
        links[link_name].pybullet_id = ix
        joints[joint_name].pybullet_id = ix




    world = {'model_id': model,
             'links': links,
             'joints': joints,
             'relations': relations}

    return world
