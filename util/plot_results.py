import argparse
import matplotlib.pyplot as plt
import numpy as np
from util import util

def plot_door_motion(plot_data):
    plt.figure()
    for data_point in plot_data:
        if data_point.mechanism_params.type == 'Door':
            plt.plot(data_point.mechanism_params.params.door_size[0], data_point.motion, 'b.')
    plt.xlabel('Door Radius')
    plt.ylabel('Motion of Handle')
    plt.title('Motion of Doors')

def plot_slider_motion(plot_data):
    plt.figure()
    for data_point in plot_data:
        if data_point.mechanism_params.type == 'Slider':
            plt.plot(data_point.mechanism_params.params.range, data_point.motion, 'b.')
    plt.xlabel('Slider Range')
    plt.ylabel('Motion of Handle')
    plt.title('Motion of Sliders')

def plot_waypoints_reached_kd(plot_data):
    pass
    #import gen.generate_policy_data
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    cm = plt.cm.get_cmap('copper')

    for data_point in plot_data:
        wr = data_point.waypoints_reached
        a = ax0.scatter([data_point.control_params.k[0]], [data_point.control_params.d[0]], \
                            cmap=cm, c=[wr], s=2, vmin=0, vmax=1) # s is markersize
        b = ax1.scatter([data_point.control_params.k[1]], [data_point.control_params.d[1]], \
                            cmap=cm, c=[wr], s=2, vmin=0, vmax=1)

    ks = np.power(10.,np.linspace(-5, 5,1000))
    gripper_mass = 1.5
    ds_critically_damped = np.sqrt(4*gripper_mass*ks)
    ax0.plot(ks, ds_critically_damped, label='critically damped')

    ax0.set_xlabel('Linear K')
    ax0.set_ylabel('Linear D')
    ax0.set_title('Time before Reached Goal or Timeout for Doors')
    ax0.legend()

    ax1.set_xlabel('Angular K')
    ax1.set_ylabel('Angular D')
    ax1.set_title('Time before Reached Goal or Timeout for Doors')

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    #ax0.set_xlim(*np.power(10.,k_lin_range))
    #ax0.set_ylim(*np.power(10.,d_lin_range))
    #ax1.set_xlim(*np.power(10.,k_rot_range))
    #ax1.set_ylim(*np.power(10.,d_rot_range))

    fig0.colorbar(a)
    fig1.colorbar(b)

def plot_results(file_name):
    print('1: Plot the door radius versus the joint motion')
    print('2: Plot the slider range versus the joint motion')
    print('3: Plot a heatmap of the # waypoints reached for varying k and d values ')
    plots = input('Above are the possible plots, type in the numbers of the plots you would like to visualize, eg. 1, 2, 3 [ENTER]\n')
    plots = list(map(int, plots.strip('[]').split(',')))

    plot_data = util.read_from_file(file_name)
    plt.ion()
    for plot in plots:
        if plot == 1:
            plot_door_motion(plot_data)
        if plot == 2:
            plot_slider_motion(plot_data)
        if plot == 3:
            plot_waypoints_reached_kd(plot_data)
    plt.show()
    input('hit [ENTER] to close plots')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fname', type=str, required=True) # give filename (without .pickle)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_results(args.fname)
