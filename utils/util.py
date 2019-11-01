import pybullet as p
import numpy as np
import pickle
import utils.transformations as trans
import math
from collections import namedtuple
import os
import torch
#from learning.models.nn_disp_pol import DistanceRegressor as NNPol
from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
#from learning.models.nn_disp_pol_mech import DistanceRegressor as NNPolMech
#from actions import policies
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

name_lookup = {'Prismatic': 0, 'Revolute': 1}
### namedtuple Definitions ###
Pose = namedtuple('Pose', 'p q')
"""
Pose represents an SE(3) pose
:param p: a vector of length 3, represeting an (x,y,z) position
:param q: a vector of length 4, representing an (x,y,z,w) quaternion
"""

Result = namedtuple('Result', 'policy_params mechanism_params net_motion cumu_motion \
                        pose_joint_world_init pose_joint_world_final config_goal \
                        image_data git_hash randomness')
"""
Result contains the performance information after the gripper tries to move a mechanism
:param policy_params: actions.policies.PolicyParams
:param mechanism_params: gen.generator_busybox.MechanismParams
:param net_motion: scalar, the net distance the mechanism handle moved, 0.0 if the gripper lost contact with the mechanism
:param net_motion: scalar, the cummulative distance the mechanism handle moved
:param pose_joint_world_init: utils.Pose object, the initial pose of the mechanism handle
:param pose_joint_world_final: utils.Pose object or None, the final pose of the mechanism handle if the
                    gripper tip is in contact with the mechanism at completion, else None
:param config_goal: the goal configuration which the joint was attempting to reach
:param image_data: utils.util.ImageData
:param git_hash: None or str representing the git hash when the data was collected
:param randomness: float in [0,1] representing how far from the true policy the random samples came from
"""

def compare_results(list_a, list_b):
    """ Compare two lists of utils.util.Results.

    Parameters:
    -----------
    list_a: list of utils.util.Results

    list_b: list of utils.util.Results

    Returns:
    --------
    same: boolean, True iff the lists are equivalent
    """

    def compare_elem(a_name, elem_a, b_name, elem_b):
        """ Compare two elements.

        Parameters:
        -----------
        a_name: str, name of elem_a

        elem_a: either a tuple, list, or something that can be checked for equality
                directly

        b_name: str, name of elem_b

        elem_b: either a tuple, list, or something that can be checked for equality
                directly

        Returns:
        --------
        boolean indicating if the elements are equivalent
        """
        ignore_names = ['git_hash']
        try:
            # if in ignore names or not tuple then this will work
            if a_name in ignore_names or (a_name not in ignore_names and elem_a == elem_b):
                return True
            else:
                print('file_1_' + a_name + ':', elem_a,
                    '\nfile_2_' + b_name + ':', elem_b)
                return False
        except:
            # enter except look if tried to check equality of a tuple
            try:
                tup_a_names = [a_name + '_' + a_field for a_field in elem_a._fields]
                tup_b_names = [b_name + '_' + b_field for b_field in elem_b._fields]
            except:
                tup_a_names = [a_name + '_' + str(a_field) for a_field in range(len(elem_a))]
                tup_b_names = [b_name + '_' + str(b_field) for b_field in range(len(elem_b))]
            same = True
            for (j, (elem_a_a, elem_b_b)) in enumerate(zip(elem_a, elem_b)):
                same = same and compare_elem(tup_a_names[j], elem_a_a, tup_b_names[j], elem_b_b)
            return same

    for (tup_a, tup_b) in zip(list_a, list_b):
        try:
            tup_a_names = tup_a._fields
            tup_b_names = tup_b._fields
        except:
            tup_a_names = range(len(tup_a))
            tup_b_names = range(len(tup_b))
        same = True
        for (i, (elem_a, elem_b)) in enumerate(zip(tup_a, tup_b)):
            same = same and compare_elem(tup_a_names[i], elem_a, tup_b_names[i], elem_b)
    return same

## Visualization Helper Functions

def viz_train_test_data(train_data, test_data):
    import matplotlib.pyplot as plt
    from gen.generator_busybox import BusyBox

    n_inter_per_bb = 100
    n_bbs = int(len(train_data)/n_inter_per_bb)
    training_dataset_size = n_bbs #10

    plt.ion()
    angle_fig, angle_ax = plt.subplots()
    pos_fig, pos_ax = plt.subplots()

    def plot_test_set():
        # plot test data angles and positions
        for (i, point) in enumerate(test_data[:10]):
            bb = BusyBox.bb_from_result(point)
            if p.getConnectionInfo()['isConnected']:
                p.disconnect()
            setup_pybullet.setup_env(bb, False, False)
            mech = bb._mechanisms[0]
            true_policy = policies.generate_policy(bb, mech, True, 0.0)
            pos = (true_policy.rigid_position[0], true_policy.rigid_position[2])
            pos_ax.plot(*pos, 'm.')
            pos_ax.annotate(i, pos)
            if i == len(test_data):
                angle_ax.plot(true_policy.pitch, 2.0, 'k.', label = 'test data')
            else:
                angle_ax.plot(true_policy.pitch, 2.0, 'k.')
            angle_ax.annotate(i, (true_policy.pitch, 2.0))

    dataset_sizes = list(range(training_dataset_size, n_bbs+1, training_dataset_size))
    pitches = []
    # plot training data angles and positions
    for n in range(1,n_bbs+1):
        point = train_data[(n-1)*n_inter_per_bb]
        if p.getConnectionInfo()['isConnected']:
            p.disconnect()
        bb = BusyBox.bb_from_result(point)
        setup_pybullet.setup_env(bb, False, False)
        mech = bb._mechanisms[0]
        true_policy = policies.generate_policy(bb, mech, True, 0.0)
        pitches += [true_policy.pitch]
        pos = (true_policy.rigid_position[0], true_policy.rigid_position[2])
        pos_ax.plot(*pos, 'c.')

        if n in dataset_sizes:
            angle_ax.clear()
            pos_ax.set_title('Positions of Training and Test Data, n_bbs='+str(n))
            pos_fig.savefig('dataset_imgs/poss_n_bbs'+str(n))

            angle_ax.set_title('Historgram of Training Data Angle, n_bbs='+str(n))
            angle_ax.hist(pitches, 30)
            plot_test_set()
            #angle_fig.savefig('dataset_imgs/pitches_n_bbs'+str(n))
            plt.show()
            input('enter to close')

            angle_ax.set_title('Historgram of Training Data Angle, n_bbs='+str(n))
            angle_ax.hist(pitches, 30)
            plot_test_set()
            #angle_fig.savefig('dataset_imgs/pitches_n_bbs'+str(n))
            plt.show()
            input('enter to close')

ImageData = namedtuple('ImageData', 'width height rgbPixels')
"""
ImageData contains a subset of the image data returned by pybullet
:param width: int, width image resolution in pixels (horizontal)
:param height: int, height image resolution in pixels (vertical)
:param rgbPixels: list of [char RED,char GREEN,char BLUE, char ALPHA] [0..width*height],
                    list of pixel colors in R,G,B,A format, in range [0..255] for each color
"""

def draw_line(endpoints, color, lifeTime=0, thick=False):
    # add to y and z dimensions
    if thick:
        n = 10
        w = .0005
        delta_ys = np.linspace(-w, w, n)
        delta_zs = np.linspace(-w, w, n)
    else:
        delta_ys = [0]
        delta_zs = [0]
    lines = []
    for delta_y in delta_ys:
        for delta_z in delta_zs:
            lines += [p.addUserDebugLine(np.add(endpoints[0], [0, delta_y, delta_z]), np.add(endpoints[1], [0, delta_y, delta_z]), color, lifeTime=lifeTime)]
    return lines

def imshow(image_data, show=True):
    img = np.reshape(image_data.rgbPixels, [image_data.height, image_data.width, 3])
    if show:
        plt.ion()
        plt.imshow(img)
        plt.show()
        input('ENTER to close plot')
    return img

### Sampling Helper Function
# TODO: want the prob of bin 0 to go to 0 as the slope increases (currently doesn't do that)
def discrete_sampler(range_vals, slope, n_bins=10):
    probs = [slope*p for p in range(1,n_bins+1)]

    # subtract diff from all the make area=1
    diff = np.subtract(1.0, sum(probs))
    diff_i = np.divide(diff, n_bins)
    probs = np.subtract(diff_i, probs)

    # then normalize (still some rounding error)
    probs = np.divide(probs, sum(probs))
    choice = np.random.choice([i for i in range(n_bins)], p=probs)
    vals = np.linspace(range_vals[0], range_vals[1], n_bins+1)
    val = np.random.uniform(vals[choice], vals[choice+1])
    return val

### Model Testing Helper Functions ###
def compare_models(model_a, model_b):
    """ Compare two pyTorch models.

    Parameters
    ----------
    model_a: a pyTorch model (loaded)

    model_b: a pyTorch model (loaded)

    Returns
    -------
    True iff the given models are identical
    """
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        if pa.data.ne(pb.data).sum() > 0:
            return False
    return True

def load_model(model_fname, hdim=32, model_type='polvis', use_cuda=False, image_encoder='spatial'):
    if model_type == 'pol':
        model = NNPol(policy_names=['Prismatic', 'Revolute'],
                      policy_dims=[2, 3],
                      hdim=hdim)
    elif model_type == 'mech':
        model = NNPolMech(policy_names=['Prismatic'],
                          policy_dims=[2],
                          mech_dims=2,
                          hdim=hdim)
    else:
        model = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                         policy_dims=[2, 3],
                         hdim=hdim,
                         im_h=53,
                         im_w=115,
                         kernel_size=3,
                         image_encoder=image_encoder)
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(model_fname, map_location=device))
    model.eval()
    if use_cuda:
        model = model.cuda()

    return model

### Writing and Reading to File Helper Functions ###
def write_to_file(file_name, data, verbose=True):
    # make directory if doesn't exist
    dir = '/'.join(file_name.split('/')[:-1])
    if not os.path.isdir(dir) and dir !='':
        os.mkdir(dir)

    # save to pickle
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle)
    if verbose:
        print('wrote dataset to '+file_name)

def read_from_file(file_name, verbose=True):
    print('reading in '+file_name)
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
        if verbose:
            print('successfully read in '+file_name)
    return data

def merge_files(in_file_names, out_file_name):
    results = []
    for file_name in in_file_names:
        result = read_from_file(file_name)
        results += result
    write_to_file(out_file_name, results)
    return results

### PyBullet Helper Functions ###

def replay_result(result):
    from actions.gripper import Gripper
    from gen.generator_busybox import BusyBox
    from utils.setup_pybullet import setup_env
    from actions.policies import get_policy_from_tuple

    bb = BusyBox.bb_from_result(result)
    image_data = setup_env(bb, True, True)
    gripper = Gripper()
    mech = bb._mechanisms[0]
    policy = get_policy_from_tuple(result.policy_params)
    pose_handle_base_world = mech.get_pose_handle_base_world()
    traj = policy.generate_trajectory(pose_handle_base_world, result.config_goal, True)
    cumu_motion, net_motion, pose_handle_world_final = gripper.execute_trajectory(traj, mech, policy.type, True)
    import pdb; pdb.set_trace()
    p.disconnect()

def vis_frame(pos, quat, length=0.2, lifeTime=0.4):
    """ This function visualizes a coordinate frame for the supplied frame where the
    red,green,blue lines correpsond to the x,y,z axes.
    :param p: a vector of length 3, position of the frame (x,y,z)
    :param q: a vector of length 4, quaternion of the frame (x,y,z,w)
    """
    new_x = transformation([length, 0.0, 0.0], pos, quat)
    new_y = transformation([0.0, length, 0.0], pos, quat)
    new_z = transformation([0.0, 0.0, length], pos, quat)

    p.addUserDebugLine(pos, new_x, [1,0,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_y, [0,1,0], lifeTime=lifeTime)
    p.addUserDebugLine(pos, new_z, [0,0,1], lifeTime=lifeTime)

def pause():
    try:
        print('press any key to continue execution')
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        print('trying to exit')
        return

### Geometric Helper Functions ###
def skew_symmetric(vec):
    """ Calculates the skew-symmetric matrix of the supplied 3 dimensional vector
    """
    i, j, k = vec
    return np.array([[0, -k, j], [k, 0, -i], [-j, i, 0]])

def adjoint_transformation(vel, translation_vec, quat, inverse=False):
    """ Converts a velocity from one frame into another
    :param vel: a vector of length 6, the velocity to be converted, [0:3] are the
            linear velocity terms and [3:6] are the angular velcoity terms
    :param translation_vec: vector of length 3, the translation (x,y,z) to the desired frame
    :param quat: vector of length 4, quaternion rotation to the desired frame
    :param inverse (optional): if True, inverts the translation_vec and quat
    """
    if inverse:
        translation_vec, quat = p.invertTransform(translation_vec, quat)

    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3,3))
    T = np.zeros((6,6))
    T[:3,:3] = R
    T[:3,3:] = skew_symmetric(translation_vec).dot(R)
    T[3:,3:] = R
    return np.dot(T, vel)

def transformation(pos, translation_vec, quat, inverse=False):
    """ Converts a position from one frame to another
    :param p: vector of length 3, position (x,y,z) in frame original frame
    :param translation_vec: vector of length 3, (x,y,z) from original frame to desired frame
    :param quat: vector of length 4, (x,y,z,w) rotation from original frame to desired frame
    :param inverse (optional): if True, inverts the translation_vec and quat
    """
    if inverse:
        translation_vec, quat = p.invertTransform(translation_vec, quat)
    R = p.getMatrixFromQuaternion(quat)
    R = np.reshape(R, (3,3))
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3] = translation_vec
    T[3,3] = 1.0
    pos = np.concatenate([pos, [1]])
    new_pos = np.dot(T, pos)
    return new_pos[:3]

def quat_math(q0, q1, inv0, inv1):
    """ Performs addition and subtraction between quaternions
    :param q0: a vector of length 4, quaternion rotation (x,y,z,w)
    :param q1: a vector of length 4, quaternion rotation (x,y,z,w)
    :param inv0: if True, inverts q0
    :param inv1: if True, inverts q1

    Examples:
    to get the total rotation from going to q0 then q1: quat_math(q0,q1,False,False)
    to get the rotation from q1 to q0: quat_math(q0,q1,True,False)
    """
    if not isinstance(q0, np.ndarray):
        q0 = np.array(q0)
    if not isinstance(q1, np.ndarray):
        q1 = np.array(q1)
    q0 = to_transquat(q0)
    q0 = trans.unit_vector(q0)
    q1 = to_transquat(q1)
    q1 = trans.unit_vector(q1)
    if inv0:
        q0 = trans.quaternion_conjugate(q0)
    if inv1:
        q1 = trans.quaternion_conjugate(q1)
    res = trans.quaternion_multiply(q0,q1)
    return to_pyquat(res)

def to_transquat(pybullet_quat):
    """Convert quaternion from (x,y,z,w) returned from pybullet to
    (w,x,y,z) convention used by transformations.py"""
    return np.concatenate([[pybullet_quat[3]], pybullet_quat[:3]])

def to_pyquat(trans_quat):
    """Convert quaternion from (w,x,y,z) returned from transformations.py to
    (x,y,z,w) convention used by pybullet"""
    return np.concatenate([trans_quat[1:], [trans_quat[0]]])

def euler_from_quaternion(q):
    """Convert quaternion from (x,y,z,w) returned from pybullet to
    euler angles = (roll, pitch, yaw) convention used by transformations.py"""
    trans_quat = to_transquat(q)
    eul = trans.euler_from_quaternion(trans_quat)
    return eul

def random_quaternion(rand=None):
    trans_quat = trans.random_quaternion(rand)
    return to_pyquat(trans_quat)

def pose_to_matrix(point, q):
    """Convert a pose to a transformation matrix
    """
    EPS = np.finfo(float).eps * 4.0
    q = to_transquat(q)
    n = np.dot(q, q)
    if n < EPS:
        M = np.identity(4)
        M[:3, 3] = point
        return M
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    M = np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])
    M[:3, 3] = point
    return M

def quaternion_from_matrix(matrix, isprecise=False):
    trans_q = trans.quaternion_from_matrix(matrix)
    return to_pyquat(trans_q)

def quaternion_from_euler(roll, pitch, yaw):
    trans_q = trans.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
    return to_pyquat(trans_q)

if __name__ == '__main__':
    train_data = read_from_file('square_bb_100.pickle')
    test_data = read_from_file('prism_gp_evals.pickle')
    viz_train_test_data(train_data, test_data)
