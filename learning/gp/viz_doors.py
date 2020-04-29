import pickle
import argparse
import numpy as np
from utils import util
from utils.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
import matplotlib.pyplot as plt
from actions.policies import Policy, Revolute, PolicyParams, get_policy_from_tuple
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from learning.dataloaders import PolicyDataset
import torch
from collections import OrderedDict

def viz_3d_plots(xs,
                 callback,
                 bb_result,
                 n_rows=2):
    """
    :param xs: A list policy tuples of form (roll, pitch, radius, q) to appear in the
        data scatterplot.
    :param callback: A callback that returns (mean, std, aq) of distance values. If
        std or aq are None, they will not be visualized.
    :param bb_result: The BusyBox being visualized.
    """
    fig = plt.figure()

    # Bin the config parameter.
    configs = np.linspace(-np.pi/2.0, 0, num=n_rows**2)
    bb = BusyBox.bb_from_result(bb_result)
    mech = bb._mechanisms[0]
    setup_env(bb, False, False, True)
    true_rad = mech.get_radius_x()
    max_dist = mech.get_max_net_motion()

    for ix, q in enumerate(configs):
        new_xs = []
        radii = np.linspace(0.08-0.025, 0.15, num=20)
        rolls = np.linspace(0, 2*np.pi, num=20)
        for ra in radii:
            for ro in rolls:
                if mech.flipped:
                    pitch = 0.0
                else:
                    pitch = np.pi
                new_xs.append([0, ro, ra, q])

        # Get the true values for this Busybox.
        ys, std, aq = callback(new_xs, bb_result)

        lookup = {}
        for jx in range(len(new_xs)):
            _, ro, ra, _ = new_xs[jx]
            if ro not in lookup:
                lookup[ro] = {}
            lookup[ro][ra] = [ys[jx], 0, 0]
            if std is not None:
                lookup[ro][ra][1] =  2*std[jx] #ys[jx]
            if aq is not None:
                lookup[ro][ra][2] = aq[jx]

        x_grid, y_grid = np.meshgrid(rolls, radii)
        z_grid = np.zeros(x_grid.shape)
        z_std_grid = np.zeros(x_grid.shape)
        z_aq_grid = np.zeros(x_grid.shape)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = lookup[x_grid[i, j]][y_grid[i, j]][0]
                z_std_grid[i, j] = lookup[x_grid[i, j]][y_grid[i, j]][1]
                z_aq_grid[i, j] = lookup[x_grid[i, j]][y_grid[i, j]][2]

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm, vmax=0.2)
        #if std is not None:
        #    ax.plot_surface(x_grid, y_grid, z_std_grid, color='r', alpha=0.5)
        if aq is not None:
            ax.plot_surface(x_grid, y_grid, z_aq_grid, color='g', alpha=0.5)

        # Plot the true radius and maximum distance.
        ax.plot([0, 2*np.pi], [true_rad, true_rad], [0, 0], c='k')
        #ax.plot_surface(x_grid, y_grid, np.ones(x_grid.shape)*max_dist, color='y', alpha=0.3)
        #ax.plot_surface(x_grid, y_grid, ys+2*std, c='r', alpha=0.5)
        # ax.plot(qs, ys + np.sqrt(BETA)*std, c='g')
        # ax.plot(qs, ys-std, c='r')
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0.08-0.025, 0.15)
        #ax.set_title('q=%.2f' % q)
        ax.set_zlim(0, 0.15)
        ax.set_xlabel('pitch')
        ax.set_ylabel('radius')
        ax.set_zlabel('dist')
    plt.savefig('doors.png')


def _true_callback(policies, bb_result):
    # Setup the BusyBox.
    bb = BusyBox.bb_from_result(bb_result)
    mech = bb._mechanisms[0]
    image_data, gripper = setup_env(bb, False, False, True)
    pose_handle_base_world = mech.get_pose_handle_base_world()

    ys = []
    for jx, (roll, pitch, radius_x, q) in enumerate(policies):
        if jx % 50 == 0:
            print(jx)
        gripper.reset(mech)
        #rot_axis_world = util.quaternion_from_euler(roll, pitch, 0.0)
        #radius = [-radius_x, 0.0, 0.0]
        #p_handle_base_world = mech.get_pose_handle_base_world().p
        #p_rot_center_world = p_handle_base_world + util.transformation(radius, [0., 0., 0.], rot_axis_world)
        #params = OrderedDict([('rot_center', p_rot_center_world),
        #                      ('rot_axis_roll', roll),
        #                      ('rot_axis_pitch', pitch),
        #                      ('rot_axis_yaw', 0.0),
        #                      ('radius_x', radius_x),
        #                      ('goal_config', q)])
        #policy_params = PolicyParams('Revolute', params, Policy.get_param_data('Revolute')) 
        #policy = get_policy_from_tuple(policy_params)
        policy = Revolute._gen(mech, x_dict={'goal_config': q,
                                             'rot_axis_yaw': 0.,
                                             'rot_axis_roll': roll,
                                             'rot_axis_pitch': pitch,
                                             'radius_x': radius_x})
        # execute
        traj = policy.generate_trajectory(pose_handle_base_world, debug=False)
        c_motion, motion, handle_pose_final = gripper.execute_trajectory(traj, mech, policy.type, False)
        gripper.reset(mech)
        ys.append(c_motion)

    return np.array(ys), None, None

def plot_true_motion(args):
    with open(args.bb_fname, 'rb') as handle:
        busybox_data = pickle.load(handle)
    busybox_data = [bb_results[0] for bb_results in busybox_data]

    for bb_result in busybox_data:
        viz_3d_plots(xs=[],
                     callback=_true_callback,
                     bb_result=bb_result)


def plot_nn_motion(args):
    model = util.load_model(args.nn_fname, args.hdim, use_cuda=True)

    def _nn_callback(policies, bb_result):
        data = []
        for roll, pitch, radius, q in policies:
            data.append({
                'type': 'Revolute',
                'params': [roll, pitch, radius],
                'config': q,
                'image': bb_result.image_data,
                'y': 0.,
                # 'mech': mech_params,
                'delta_vals': [0, 0, 0]
            })

        dataset = PolicyDataset(data)
        ys = []
        for i in range(len(dataset.items)):
            policy_type = dataset.items[i]['type']
            policy_type_tensor = torch.Tensor([util.name_lookup[policy_type]])
            policy_tensor = dataset.tensors[i].unsqueeze(0)
            config_tensor = dataset.configs[i].unsqueeze(0)
            image_tensor = dataset.images[i].unsqueeze(0)
            if True:
                policy_type_tensor = policy_type_tensor.cuda()
                policy_tensor = policy_tensor.cuda()
                config_tensor = config_tensor.cuda()
                image_tensor = image_tensor.cuda()
            pred_motion, _ = model.forward(policy_type_tensor,
                                           policy_tensor,
                                           config_tensor,
                                           image_tensor)
            if True:
                pred_motion_float = pred_motion.cpu().detach().numpy()[0][0]
            else:
                pred_motion_float = pred_motion.detach().numpy()[0][0]
            ys += [pred_motion_float]

        return ys, None, None

    with open(args.bb_fname, 'rb') as handle:
        busybox_data = pickle.load(handle)
    busybox_data = [bb_results[0] for bb_results in busybox_data]

    for bb_result in busybox_data:

        viz_3d_plots(xs=[],
                     callback=_nn_callback,
                     bb_result=bb_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--L',
        type=int,
        required=True,
        help='what number of training Mechanisms to evaluate')
    parser.add_argument(
        '--n-bbs',
        type=int,
        help='number of BusyBoxes to visualize')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--hdim',
        type=int,
        default=16,
        help='hdim of supplied model(s), used to load model file')
    parser.add_argument(
        '--nn-fname',
        help='path to model files')
    parser.add_argument(
        '--use-gripper',
        help='use to apply foce directly to handles')
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    plot_nn_motion(args)
