import argparse
import matplotlib.pyplot as plt
import numpy as np
from util import util
import sys

class PlotFunc(object):

    @staticmethod
    def description():
        return 'No description for PlotFunc: ' + self.__name__

    def plot(self, figure_num, plot_data):
        plt.figure(figure_num)
        self._plot(plot_data)

class DoorRadiusMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the door radius versus the joint motion'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Door':
                plt.plot(data_point.mechanism_params.params.door_size[0], data_point.motion, 'b.')
        plt.xlabel('Door Radius')
        plt.ylabel('Motion of Handle')
        plt.title('Motion of Doors')

class SliderRangeMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the slider range versus the joint motion'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Slider':
                plt.plot(data_point.mechanism_params.params.range, data_point.motion, 'b.')
        plt.xlabel('Slider Range')
        plt.ylabel('Motion of Handle')
        plt.title('Motion of Sliders')

class DoorRadiusWR(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the door radius versus the % of waypoints reached'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Door':
                plt.plot(data_point.mechanism_params.params.door_size[0], data_point.waypoints_reached, 'b.')
        plt.xlabel('Door Radius')
        plt.ylabel('Percentage of Trajectory Waypoints Reached')
        plt.title('Waypoints Reached by Door')

class SliderRangeWR(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the slider range versus the % of waypoints reached'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Slider':
                plt.plot(data_point.mechanism_params.params.range, data_point.waypoints_reached, 'b.')
        plt.xlabel('Slider Range')
        plt.ylabel('Percentage of Trajectory Waypoints Reached')
        plt.title('Slider Range vs Waypoints Reached')

class SliderAxisMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the slider axis versus the joint_motion'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Slider':
                angle = np.arccos(data_point.mechanism_params.params.axis[0])
                plt.plot(angle, data_point.motion, 'b.')
        plt.xlabel('Slider Axis Angle')
        plt.ylabel('Percentage of Trajectory Waypoints Reached')
        plt.title('Motion of Handle')

class SliderAxisWR(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the slider axis versus the % of waypoints reached'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.mechanism_params.type == 'Slider':
                angle = np.arccos(data_point.mechanism_params.params.axis[0])
                plt.plot(angle, data_point.waypoints_reached, 'b.')
        plt.xlabel('Slider Axis Angle')
        plt.ylabel('Percentage of Trajectory Waypoints Reached')
        plt.title('Slider Angle vs Waypoints Reached')

class WRKD(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a heatmap of the % waypoints reached for varying k and d values'

    def _plot(self, plot_data):
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

def print_stats(data):
    stats = {}
    for data_point in data:
        mech_type = data_point.mechanism_params.type
        policy_type = data_point.policy_params.type
        key = (mech_type, policy_type)
        if key not in stats:
            stats[key] = 1
        else:
            stats[key] += 1
    print('Stats on the dataset')
    for (key, val) in stats.items():
        sys.stdout.write('  %s mech, %s policy: %i\n' % (*key, val))

def plot_results(file_name):
    data = util.read_from_file(file_name)
    print_stats(data)

    plot_funcs = PlotFunc.__subclasses__()
    for (i, func) in enumerate(plot_funcs):
        print(i, ':', func.description())
    plot_nums = input('Above are the possible plots, type in the numbers of the plots you would like to visualize, eg. 1, 3, 4 [ENTER]\n')
    plot_nums = list(map(int, plot_nums.strip('[]').split(',')))

    plt.ion()
    for plot_num in plot_nums:
        plot_func = plot_funcs[plot_num]()
        plot_func.plot(plot_num, data)
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
