import pybullet as p
import numpy as np
import pickle

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


def vis_frame(pos, orn, rgb=[1,0,0], length=.2, life=.4):
    new_x = transformation([0, 0, length], pos, orn)
    new_y = transformation([0, length, 0], pos, orn)
    new_z = transformation([length, 0, 0], pos, orn)

    p.addUserDebugLine(pos, new_x, rgb, lifeTime=life)
    p.addUserDebugLine(pos, new_y, rgb, lifeTime=life)
    p.addUserDebugLine(pos, new_z, rgb, lifeTime=life)

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
