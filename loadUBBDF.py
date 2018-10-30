import xml.etree.ElementTree as ET
from ubbdf_defs import Link, Joint, Relation

def loadUBBDF(urdf_file, ubbdf_file):
    # get links and joints from urdf
    urdf_tree = ET.parse('models/busybox/model.urdf')
    urdf_root = urdf_tree.getroot()

    links = []
    for link in urdf_root.findall('link'):
        name = link.attrib['name']
        links += [Link(name)]

    joints = []
    for joint in urdf_root.findall('joint'):
        name = joint.attrib['name']
        type = joint.attrib['type']
        for j_elem in joint:
            if j_elem.tag == 'child':
                child_link = j_elem.attrib['link']
        joints += [Joint(name, type, child_link)]

    # get relations from ubbdf
    ubbdf_tree = ET.parse('models/busybox/model_relations.ubbdf')
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

    return links, joints, relations
