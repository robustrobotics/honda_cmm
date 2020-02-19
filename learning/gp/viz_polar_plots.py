import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import itertools
from functools import reduce
from actions.policies import Policy, PolicyParams
from utils import util
from learning.dataloaders import PolicyDataset, parse_pickle_file
from collections import namedtuple
from actions.policies import Prismatic, Revolute
from gen.generate_policy_data import get_true_ys

PlotData = namedtuple('PlotData', 'param_name varied range')
N_BINS = 5

n_angular = 40
n_linear = 20 # NOTE: Must be an even number!!!

def viz_circles(plot_mode, image_data, mech, beta=None, sample_points={}, opt_points=[], \
                gps=None, nn=None, bb_i=0, plot_dir_prefix=''):
    # make figure of an image of the mechanism
    plt.ion()
    fig, ax = plt.subplots()
    w, h, im = image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax.imshow(np_im)
    policy_types = ['Prismatic', 'Revolute']
    mean_figs = []
    all_mean_axes = []

    for policy_type in policy_types:
        if plot_mode == util.GP_PLOT or plot_mode == util.GP_NN_PLOT:
            gp = gps[policy_type]
        all_param_data = Policy.get_param_data(policy_type)

        n_params = len([name for name, param_data in all_param_data.items() if param_data.varied])

        all_angular_params = [PlotData(param_name, param_data.varied, param_data.bounds)
                                for (param_name, param_data) in all_param_data.items()
                                if param_data.type =='angular']
        all_linear_params = [PlotData(param_name, param_data.varied, param_data.bounds)
                                for (param_name, param_data) in all_param_data.items()
                                if param_data.type =='linear']

        # for each pair of (linear, angular) param pairs make a figure
        for angular_param in all_angular_params:
            if not angular_param.varied:
                continue
            for linear_param in all_linear_params:
                if not linear_param.varied:
                    continue
                linear_vals = np.linspace(*linear_param.range, n_linear)
                angular_vals = np.linspace(*angular_param.range, n_angular)
                l, a = np.meshgrid(linear_vals, angular_vals)
                mean_colors = np.zeros(l.shape)

                # TODO: only works for up to 2 other_params (will have to figure out new
                # visualization past that)
                mean_fig = plt.figure()
                mean_figs.append(mean_fig)
                mean_fig_axes = []
                plt.suptitle(policy_type + ' mean fn:' + angular_param.param_name + \
                                ' vs ' + linear_param.param_name)

                std_fig = plt.figure()
                plt.suptitle(policy_type + ' std:' + angular_param.param_name + \
                                ' vs ' + linear_param.param_name)

                ucb_fig = plt.figure()
                plt.suptitle(policy_type + ' ucb:' + angular_param.param_name + \
                                ' vs ' + linear_param.param_name)

                # bin the other param values
                all_other_params = [PlotData(param_name, param_data.varied, param_data.bounds)
                                        for (param_name, param_data) in all_param_data.items()
                                        if param_name != angular_param.param_name
                                            and param_name != linear_param.param_name
                                            and param_data.varied]

                subplot_inds_and_vals = {}
                for other_param in all_other_params:
                    subplot_inds_and_vals[other_param.param_name] = \
                        list(enumerate(np.linspace(*other_param.range, N_BINS)))

                if len(subplot_inds_and_vals) == 0:
                    n_rows = 1
                    n_cols = 1
                else:
                    keys = list(subplot_inds_and_vals.keys())
                    if len(subplot_inds_and_vals) == 1:
                        n_rows = 1
                        n_cols = len(subplot_inds_and_vals[keys[0]])
                    elif len(subplot_inds_and_vals) == 2:
                        n_rows = len(subplot_inds_and_vals[keys[0]])
                        n_cols = len(subplot_inds_and_vals[keys[1]])

                # for each other value add a dimension of plots to the figure
                for single_subplot_inds_and_vals in itertools.product(*subplot_inds_and_vals.values()):
                    # make matrix of all values to predict dist for
                    x_inds = {}
                    X_pred = np.zeros((n_angular*n_linear, n_params))
                    xpred_rowi = 0
                    for ix in range(0, n_angular):
                        for jx in range(0, n_linear):
                            x = []
                            for param_name, param_data in all_param_data.items():
                                if param_data.varied:
                                    if param_name == angular_param.param_name:
                                        x.append(angular_vals[ix])
                                    elif param_name == linear_param.param_name:
                                        x.append(linear_vals[jx])
                                    else:
                                        for name, (ind, val) in \
                                                zip(subplot_inds_and_vals, \
                                                    single_subplot_inds_and_vals):
                                            if param_name == name:
                                                x.append(val)
                                                x_inds[name] = len(x) - 1
                            X_pred[xpred_rowi,:] = x
                            xpred_rowi += 1

                    Y_pred = np.zeros((X_pred.shape[0]))
                    if plot_mode == util.GP_PLOT:
                        Y_pred_gp, Y_std = gp.predict(X_pred, return_std=True)
                        Y_pred = np.add(Y_pred, Y_pred_gp.squeeze())
                        Y_std = Y_std.squeeze()
                        std_colors = Y_std.reshape(n_angular, n_linear)
                    if plot_mode == util.GP_NN_PLOT:
                        loader = format_batch(policy_type, X_pred, mech, image_data)
                        k, x, im, _, _ = next(iter(loader))
                        pol = torch.Tensor([util.name_lookup[k[0]]])
                        nn_preds = nn(pol, x.float(), im)[0].detach().numpy()
                        Y_pred = np.add(Y_pred, nn_preds.squeeze())
                    if plot_mode == util.GROUND_TRUTH_PLOT:
                        Y_pred = get_true_ys(X_pred, mech, \
                                    PolicyParams(policy_type, None, all_param_data))
                    mean_colors = Y_pred.reshape(n_angular, n_linear)

                    if plot_mode == util.GP_PLOT:
                        ucb = np.add(Y_pred, np.sqrt(beta) * Y_std)
                        ucb_colors = ucb.reshape(n_angular, n_linear)

                    if len(single_subplot_inds_and_vals) == 0:
                        subplot_num = 1
                    else:
                        row_col = [o[0]+1 for o in single_subplot_inds_and_vals]
                        if len(row_col) == 1:
                            subplot_num = 1*row_col[0]
                        else:
                            subplot_num = reduce(lambda x, y: n_cols*(x-1)+y, row_col)

                    # make polar subplot of mean function
                    ax = mean_fig.add_subplot(n_rows, n_cols, subplot_num, projection='polar')
                    mean_fig_axes.append(ax)
                    ax.set_title('\n'.join([str(all_other_params[other_param_i].param_name)
                        + ' = ' + str("%.2f" % other_val) for other_param_i,
                        (subplot_i, other_val) in enumerate(single_subplot_inds_and_vals)]),
                        fontsize=10)
                    max_dist = mech.get_max_net_motion()

                    # only add points to subplot that are close to this subplot "bin"
                    plot_points = []
                    pt_colors = sample_points[policy_type]
                    if not opt_points == [] and opt_points[0] == policy_type:
                        #TODO: something on this line is broken
                        pt_colors += opt_points[1]
                    if subplot_inds_and_vals == {}:
                        plot_points = [(get_plot_point(x,
                                                        angular_param.param_name,
                                                        linear_param.param_name,
                                                        all_param_data, policy_type), c)
                                                            for (x, c) in pt_colors]
                    else:
                        for pt, color in pt_colors:
                            keep_point = True
                            for other_param_name, (subplot_dim, subplot_val) in  \
                                    zip(subplot_inds_and_vals, single_subplot_inds_and_vals):
                                all_subplot_vals = np.array([inds_and_vals[1] for
                                        inds_and_vals in subplot_inds_and_vals[other_param_name]])
                                point_param_val = pt[x_inds[other_param_name]]
                                closest_index = (np.abs(all_subplot_vals - point_param_val)).argmin()
                                if all_subplot_vals[closest_index] == subplot_val:
                                    keep_point = keep_point and True
                                else:
                                    keep_point = keep_point and False
                            if keep_point:
                                plot_points.append((get_plot_point(pt,
                                                            angular_param.param_name,
                                                            linear_param.param_name,
                                                            all_param_data, policy_type), color))

                    mean_im = polar_plots(ax, mean_colors, max_dist, angular_param,
                                     linear_param, points=plot_points)

                    if plot_mode == util.GP_PLOT or plot_mode == util.GP_NN_PLOT:
                        # make polar subplot of std dev
                        ax = std_fig.add_subplot(n_rows, n_cols, subplot_num, projection='polar')
                        ax.set_title('\n'.join([str(all_other_params[other_param_i].param_name)
                            + ' = ' + str("%.2f" % other_val) for other_param_i,
                            (subplot_i, other_val) in enumerate(single_subplot_inds_and_vals)]),
                            fontsize=10)
                        max_dist = mech.get_max_net_motion()
                        std_im = polar_plots(ax, std_colors, max_dist, angular_param,
                                         linear_param, points=plot_points)

                        # name polar plot for ucb criteria
                        ax = ucb_fig.add_subplot(n_rows, n_cols, subplot_num, projection='polar')
                        ax.set_title('\n'.join([str(all_other_params[other_param_i].param_name)
                            + ' = ' + str("%.2f" % other_val) for other_param_i,
                            (subplot_i, other_val) in enumerate(single_subplot_inds_and_vals)]),
                            fontsize=10)
                        max_dist = mech.get_max_net_motion()
                        ucb_im = polar_plots(ax, ucb_colors, max_dist, angular_param,
                                         linear_param, points=plot_points)

                add_colorbar(mean_fig, mean_im)
                plot_list = [('mean', mean_fig)]
                all_mean_axes.append(mean_fig_axes)

                if plot_mode == util.GP_PLOT or plot_mode == util.GP_NN_PLOT:
                    add_colorbar(std_fig, std_im)
                    plot_list.append(('std_dev', std_fig))
                    add_colorbar(ucb_fig, ucb_im)
                    plot_list.append(('ucb', ucb_fig))

                for plot_type, fig in plot_list:
                    # folder for all plot figures
                    plot_dir = 'plots/'
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    # (optionally) another folder set with --plot-dir
                    if plot_dir_prefix is not '':
                        plot_dir += plot_dir_prefix
                        if not os.path.isdir(plot_dir):
                            os.mkdir(plot_dir)
                    # folder for bb number
                    plot_dir += '/bb_%i' % bb_i
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    # folder for each policy type
                    plot_dir += '/'+policy_type
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    # folder for each policy combination
                    plot_dir += '/%s' % angular_param.param_name+linear_param.param_name
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    # folder for each plot type (mean fn and std dev)
                    plot_dir += '/%s' % plot_type
                    if not os.path.isdir(plot_dir):
                        os.mkdir(plot_dir)
                    # file name is the interaction number
                    sample_num = sum([len(sample_points[pt]) for pt in sample_points])
                    fig.savefig(plot_dir+'/%i.png' % sample_num)
    util.write_to_file('mean_fig_door.pickle', [mean_figs, all_mean_axes, \
                        all_param_data, all_angular_params, all_linear_params])

def polar_plots(ax, colors, vmax, angular_param, linear_param, points=None):
    n_ang, n_lin = colors.shape

    # for prismatic policies
    if (abs(angular_param.range[0]-angular_param.range[1]) == np.pi) and \
       (linear_param.range[0] == -linear_param.range[1]):
        thp = np.linspace(0, 2*np.pi, n_ang*2)
        rp = np.linspace(0, max(linear_param.range), n_lin//2)
        rp, thp = np.meshgrid(rp, thp)
        cp = np.zeros(rp.shape)

        if abs(angular_param.range[0]) == 0 or abs(angular_param.range[0]) == np.pi:
            cp[0:n_ang, :] = colors[:, 0:n_lin//2][:, ::-1]
            cp[n_ang:, :] = np.copy(colors[:, n_lin//2:])
        elif abs(angular_param.range[0]) == np.pi/2:
            cp[0:n_ang//2, :] = np.copy(colors[n_ang//2:, n_lin//2:])
            cp[n_ang//2:n_ang, :] = colors[:n_ang//2, :n_lin//2][:, ::-1]
            cp[n_ang:3*n_ang//2, :] = colors[n_ang//2:, :n_lin//2][:, ::-1]
            cp[3*n_ang//2:2*n_ang, :] = np.copy(colors[:n_ang//2, n_lin//2:])
        type = 'Prismatic'
    # for revolute policies
    else:
        thp = np.linspace(*angular_param.range, n_ang)
        rp = np.linspace(*linear_param.range, n_lin)
        rp, thp = np.meshgrid(rp, thp)
        cp = colors
        type = 'Revolute'
    cbar = ax.pcolormesh(thp, rp, cp, vmin=0, vmax=vmax, cmap='viridis')
    ax.tick_params(axis='x', colors='white')
    ax.set_yticklabels([])
    add_points(ax, points)

    return cbar

def add_colorbar(fig, im):
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def format_batch(ptype, x_pred, mech, image_data):

    parsed_data = []
    for ix in range(x_pred.shape[0]):
        parsed_data.append({
            'type': ptype,
            'params': x_pred[ix],
            'image': image_data,
            'y': 0.0
        })
    dataset = PolicyDataset(parsed_data)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=len(dataset))
    return train_loader


# takes in an x and returns the values to be plotted
def get_plot_point(x, angular_name, linear_name, all_param_data, policy_type):
    if policy_type == 'Prismatic':
        flipped = False
        xi = 0
        for (param_name, param_data) in all_param_data.items():
            if param_data.varied:
                if param_name == 'goal_config':
                    if x[xi] < 0.0:
                        flipped = True
                xi += 1

    xi = 0
    point = []
    for (param_name, param_data) in all_param_data.items():
        if param_data.varied:
            if param_name == angular_name or param_name == linear_name:
                if policy_type == 'Prismatic' and flipped:
                    if param_name == 'pitch':
                        point.append(x[xi] - np.pi)
                    if param_name == 'goal_config':
                        point.append(-1*x[xi])
                else:
                    point.append(x[xi])
            xi += 1
    return point

def add_points(ax, points):
    if not points is None:
        for (point, c) in points:
            ax.scatter(*point, c=c, s=3)

def add_points_to_saved_ax(explr_p=1.0, sample_points={}):
    fname = 'mean_fig_door.pickle'
    figs, axes, all_param_data, all_angular_params, all_linear_params = util.read_from_file(fname)
    figs = figs[1:]
    axes = axes[1:]
    curr_fig = 0
    for policy_type in ['Revolute']:#'Prismatic'
        n_params = len([name for name, param_data in all_param_data.items() if param_data.varied])
        for angular_param in all_angular_params:
            if not angular_param.varied:
                continue
            for linear_param in all_linear_params:
                if not linear_param.varied:
                    continue

                linear_vals = np.linspace(*linear_param.range, n_linear)
                angular_vals = np.linspace(*angular_param.range, n_angular)
                l, a = np.meshgrid(linear_vals, angular_vals)

                # TODO: only works for up to 2 other_params (will have to figure out new
                # visualization past that)

                plt.suptitle(policy_type + ' mean fn:' + angular_param.param_name + \
                                ' vs ' + linear_param.param_name)

                # bin the other param values
                all_other_params = [PlotData(param_name, param_data.varied, param_data.bounds)
                                        for (param_name, param_data) in all_param_data.items()
                                        if param_name != angular_param.param_name
                                            and param_name != linear_param.param_name
                                            and param_data.varied]

                subplot_inds_and_vals = {}
                for other_param in all_other_params:
                    subplot_inds_and_vals[other_param.param_name] = \
                        list(enumerate(np.linspace(*other_param.range, N_BINS)))

                if len(subplot_inds_and_vals) == 0:
                    n_rows = 1
                    n_cols = 1
                else:
                    keys = list(subplot_inds_and_vals.keys())
                    if len(subplot_inds_and_vals) == 1:
                        n_rows = 1
                        n_cols = len(subplot_inds_and_vals[keys[0]])
                    elif len(subplot_inds_and_vals) == 2:
                        n_rows = len(subplot_inds_and_vals[keys[0]])
                        n_cols = len(subplot_inds_and_vals[keys[1]])

                curr_ax_i = 0
                for single_subplot_inds_and_vals in itertools.product(*subplot_inds_and_vals.values()):
                    x_inds = {}
                    X_pred = np.zeros((n_angular*n_linear, n_params))
                    xpred_rowi = 0
                    for ix in range(0, n_angular):
                        for jx in range(0, n_linear):
                            x = []
                            for param_name, param_data in all_param_data.items():
                                if param_data.varied:
                                    if param_name == angular_param.param_name:
                                        x.append(angular_vals[ix])
                                    elif param_name == linear_param.param_name:
                                        x.append(linear_vals[jx])
                                    else:
                                        for name, (ind, val) in \
                                                zip(subplot_inds_and_vals, \
                                                    single_subplot_inds_and_vals):
                                            if param_name == name:
                                                x.append(val)
                                                x_inds[name] = len(x) - 1
                            X_pred[xpred_rowi,:] = x
                            xpred_rowi += 1

                    if len(single_subplot_inds_and_vals) == 0:
                        subplot_num = 1
                    else:
                        row_col = [o[0]+1 for o in single_subplot_inds_and_vals]
                        if len(row_col) == 1:
                            subplot_num = 1*row_col[0]
                        else:
                            subplot_num = reduce(lambda x, y: n_cols*(x-1)+y, row_col)

                    # make polar subplot of mean function
                    axes[curr_fig][curr_ax_i].set_title('\n'.join([str(all_other_params[other_param_i].param_name)
                        + ' = ' + str("%.2f" % other_val) for other_param_i,
                        (subplot_i, other_val) in enumerate(single_subplot_inds_and_vals)]),
                        fontsize=10)

                    # only add points to subplot that are close to this subplot "bin"
                    plot_points = []
                    pt_colors = sample_points[policy_type]
                    if subplot_inds_and_vals == {}:
                        plot_points = [(get_plot_point(x,
                                                        angular_param.param_name,
                                                        linear_param.param_name,
                                                        all_param_data, policy_type), c)
                                                            for (x, c) in pt_colors]
                    else:
                        for pt, color in pt_colors:
                            keep_point = True
                            for other_param_name, (subplot_dim, subplot_val) in  \
                                    zip(subplot_inds_and_vals, single_subplot_inds_and_vals):
                                all_subplot_vals = np.array([inds_and_vals[1] for
                                        inds_and_vals in subplot_inds_and_vals[other_param_name]])
                                point_param_val = pt[x_inds[other_param_name]]
                                closest_index = (np.abs(all_subplot_vals - point_param_val)).argmin()
                                if all_subplot_vals[closest_index] == subplot_val:
                                    keep_point = keep_point and True
                                else:
                                    keep_point = keep_point and False
                            if keep_point: plot_points.append((get_plot_point(pt,
                                                                angular_param.param_name,
                                                                linear_param.param_name,
                                                                all_param_data, policy_type), color))
                    #print(len(axes), len(axes[0]), curr_fig, curr_ax_i)
                    add_points(axes[curr_fig][curr_ax_i], plot_points)

                    curr_ax_i += 1
                curr_fig += 1

    #plt.show()
    for fi, fig in enumerate(figs):
        fig.savefig('voo_testing/gpucb/omega_%f_%i_doors.png' % (explr_p, fi))

if __name__ == '__main__':
    add_points_to_saved_ax()
