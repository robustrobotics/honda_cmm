import pybullet as p
import numpy as np
import pickle
import transformations as trans

class Recorder(object):

    def __init__(self, height, width):
        self.frames = []
        self.height = height
        self.width = width

    def capture(self):
        h, w, rgb, depth, seg = p.getCameraImage(self.width, self.height)
        self.frames.append({'rgb': rgb,
                            'depth': depth})

    def save(self, fname):
        with open(fname, 'wb') as handle:
            pickle.dump(self.frames, handle)


def vis_frame(pos, orn, length=.2, lifeTime=.4):
    new_x = transformation([length, 0, 0], pos, orn)
    new_y = transformation([0, length, 0], pos, orn)
    new_z = transformation([0, 0, length], pos, orn)

    p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)

def transformation(pos, translation_vec, quat):
    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3,3))
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3] = translation_vec
    T[3,3] = 1
    pos = np.concatenate([pos, [1]])
    new_pos = np.dot(T, pos)
    return new_pos[:3]

def diff_quat(q0, q1, step_size=None, add=False):
    q0 = to_transquat(q0)
    q0 = trans.unit_vector(q0)
    q1 = to_transquat(q1)
    q1 = trans.unit_vector(q1)
    if add:
        q1 = trans.quaternion_conjugate(q1)
    orn_err = trans.quaternion_multiply(q0, trans.quaternion_conjugate(q1))
    orn_err = to_pyquat(orn_err)

    orn_des = None
    if step_size is not None:
        orn_des = trans.quaternion_slerp(q1, q0, step_size)
        orn_des = to_pyquat(orn_des)
    return orn_err, orn_des

def to_transquat(pybullet_quat):
    return np.concatenate([[pybullet_quat[3]], pybullet_quat[:3]])

def to_pyquat(trans_quat):
    return np.concatenate([trans_quat[1:], [trans_quat[0]]])

def euler_from_quaternion(q):
    trans_quat = to_transquat(q)
    eul = trans.euler_from_quaternion(trans_quat)
    return eul
