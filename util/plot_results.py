import argparse
import matplotlib.pyplot as plt
import numpy as np
from util import util
import sys
from learning.test_model import SearchResult, SampleResult
from actions.policies import PrismaticParams, RevoluteParams, get_policy_from_params
from gen.generator_busybox import BusyBox, Slider, Door
from gen.generate_policy_data import generate_samples
from learning.dataloaders import parse_pickle_file, PolicyDataset
import actions.policies as policies

class PlotFunc(object):

    @staticmethod
    def description():
        return 'No description for PlotFunc: ' + self.__name__

    def plot(self, figure_num, plot_data):
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

class SliderConfigMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the goal config percentage versus the joint motion for a slider'

    def _plot(self, plot_data):
        plt.figure()
        for data_point in plot_data:
            percentage = data_point.config_goal/(data_point.mechanism_params.params.range/2)
            plt.plot(percentage, data_point.net_motion, 'b.')
        print(plot_data[0].mechanism_params.params.range/2)
        plt.xlabel('Goal Config (%)')
        plt.ylabel('Motion of Handle')
        plt.title('NET Motion of Sliders')

        plt.figure()
        for data_point in plot_data:
            percentage = data_point.config_goal/(data_point.mechanism_params.params.range/2)
            plt.plot(percentage, data_point.cumu_motion, 'b.')
        plt.xlabel('Goal Config (%)')
        plt.ylabel('Motion of Handle')
        plt.title('CUMMULATIVE Motion of Sliders')

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
        plt.ylabel('Motion of Handle')
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

class MotionKD(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a heatmap of the motion reached for varying k and d values (fixed q)'

    def _plot(self, plot_data):
        fig, ax = plt.subplots()
        cm = plt.cm.get_cmap('viridis')

        goal_config = plot_data[0].result.config_goal
        ks = [p.k for p in plot_data]
        ds = [p.d for p in plot_data]
        motions = [p.result.net_motion for p in plot_data]
        a = ax.scatter(ks, ds, c=motions, cmap=cm, s=4)
        #for data_point in plot_data:
            #if data_point.result.net_motion > .07 and data_point.result.net_motion < .08:
        #    a = ax.scatter([data_point.k], [data_point.d], c=[data_point.result.net_motion],
        #            cmap=cm, s=4)#, vmin=0.07, vmax=0.08)
                #print(data_point.result.net_motion)
        ax.set_xlabel('Linear K')
        ax.set_ylabel('Linear D')
        ax.set_title('Motion Generated, q_{goal}='+str(goal_config))
        #ax.legend()
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        #ax.set_xlim(*np.power(10.,[2,6]))
        #ax.set_ylim(*np.power(10.,[-5,5]))
        fig.colorbar(a)

class MechanismMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a histogram of the motion generated for each (mechanism type, policy tried) in a dataset'

    def _plot(self, plot_data):
        plt_cont = input('Only plot motion if gripper touching handle at end of execution? [y/n] then [ENTER]')
        data_hist = {}
        for data_point in plot_data:
            key = data_point.mechanism_params.type + ', ' +  data_point.policy_params.type
            if plt_cont == 'n' or (plt_cont == 'y' and data_point.pose_joint_world_final):
                if key in data_hist:
                    data_hist[key].append(data_point.motion)
                else:
                    data_hist[key] = []

        colors = {'Slider, Prismatic': 'blue', 'Slider, Revolute': 'orange', \
                    'Door, Prismatic': 'green', 'Door, Revolute': 'red'}
        ordered_colors = [colors[key] for key in data_hist.keys()]
        plt.hist(data_hist.values(), 20, histtype='bar', label=data_hist.keys(), color=ordered_colors)
        plt.xlabel('Motion')
        plt.ylabel('Frequency')
        plt.title('Motion of Mechanisms')
        plt.legend()

class MotionRandomness(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the randomness in a policy versus the motion generated'

    def _plot(self, plot_data):
        for data_point in plot_data:
            if data_point.pose_joint_world_final is None:
                plt.plot(data_point.randomness, data_point.motion, 'b.')
            else:
                plt.plot(data_point.randomness, data_point.motion, 'c.')
        plt.xlabel('Randomness')
        plt.ylabel('Motion')
        plt.title('Randomness versus Motion for a '+data_point.mechanism_params.type)

class SliderPolicyDelta(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the motion generated versus to distance from the true slider policy'

    def _plot(self, plot_data):
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        cm = plt.cm.get_cmap('copper')

        fig = plt.figure()
        ax = Axes3D(fig)
        delta_yaws = [data_point.policy_params.delta_values.delta_yaw for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        delta_pitches = [data_point.policy_params.delta_values.delta_pitch for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        goal_configs =  [data_point.config_goal for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        motion = [data_point.motion for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        im = ax.scatter(delta_yaws, delta_pitches, goal_configs, c=motion, cmap=cm, vmin=min(motion), vmax=max(motion))
        fig.colorbar(im)
        ax.set_xlabel('Delta Yaw')
        ax.set_ylabel('Delta Pitch')
        ax.set_title('Motion for Varying Prismatic Policy Values')

class DoorPolicyDelta(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the motion generated versus to distance from the true door policy'

    def _plot(self, plot_data):
        plt.figure()
        for data_point in plot_data:
            if data_point.pose_joint_world_final is None:
                c = 'b.'
            else:
                c = 'c.'
            plt.plot(data_point.delta_values.delta_roll, data_point.motion, c)
        plt.xlabel('Delta Axis Roll')
        plt.ylabel('Motion')
        plt.title('Distance from true axis roll value versus Motion for a '+data_point.mechanism_params.type)

        plt.figure()
        for data_point in plot_data:
            if data_point.pose_joint_world_final is None:
                c = 'b.'
            else:
                c = 'c.'
            plt.plot(data_point.delta_values.delta_pitch, data_point.motion, c)
        plt.xlabel('Delta Axis Pitch')
        plt.ylabel('Motion')
        plt.title('Distance from true axis pitch value versus Motion for a '+data_point.mechanism_params.type)

        plt.figure()
        for data_point in plot_data:
            if data_point.pose_joint_world_final is None:
                c = 'b.'
            else:
                c = 'c.'
            plt.plot(data_point.delta_values.delta_radius_x, data_point.motion, c)
        plt.xlabel('Delta Radius x')
        plt.ylabel('Motion')
        plt.title('Distance from true radius x value versus Motion for a '+data_point.mechanism_params.type)

        plt.figure()
        for data_point in plot_data:
            if data_point.pose_joint_world_final is None:
                c = 'b.'
            else:
                c = 'c.'
            plt.plot(data_point.delta_values.delta_radius_z, data_point.motion, c)
        plt.xlabel('Delta Radius z')
        plt.ylabel('Motion')
        plt.title('Distance from true radius z value versus Motion for a '+data_point.mechanism_params.type)

class YawPitchMotionResults(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a heatmap of the predicted motion for yaw versus pitch for several q values 5 mechanism of 1000 results each'

    def _plot(self, data):
        for m in range(1,6):
            mech_data = data[(m-1)*1000:m*1000]
            config_data = {}
            for point in mech_data:
                if point.config_goal not in config_data:
                    config_data[point.config_goal] = [[] for _ in range(3)]
            limit = mech_data[0].mechanism_params.params.range/2
            for point in mech_data:
                config = point.config_goal
                if point.pose_joint_world_final is not None:
                    config_data[config][0] += [point.policy_params.delta_values.delta_yaw]
                    config_data[config][1] += [point.policy_params.delta_values.delta_pitch]
                    config_data[config][2] += [point.cumu_motion]
                '''
                if round(config,2) == -.03 or round(config,2) == .03:
                    if abs(point.policy_params.delta_values.delta_yaw) < .1:
                        pitch = point.policy_params.delta_values.delta_pitch
                        if abs(pitch) < .4 and abs(pitch) > .3:
                            if point.net_motion > 0.0:
                                print(point.policy_params.delta_values.delta_yaw)
                                print(point.policy_params.delta_values.delta_pitch)
                                util.replay_result(point)
                #else:
                #    util.replay_result(point)
                '''
            min_motion = min([min(config_data[config][2]) for config in config_data.keys()])
            max_motion = max([max(config_data[config][2]) for config in config_data.keys()])

            configs = list(config_data.keys())
            configs.sort()
            n_configs = len(configs)
            config_num = 0
            lw = int(round(np.sqrt(n_configs)))
            fig, axes = plt.subplots(lw, lw, sharex=True, sharey=True)
            plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
            for ax in axes.flatten():
                if config_num < n_configs:
                    im = ax.scatter(config_data[configs[config_num]][0],
                                    config_data[configs[config_num]][1],
                                    c=config_data[configs[config_num]][2],
                                    vmin=min_motion, vmax=max_motion)
                    ax.set_title(str(round(configs[config_num], 2))+', limit='+str(round(limit, 2)))
                    config_num += 1
            im.set_clim(min_motion, max_motion)
            plt.xlabel('Delta Yaw')
            plt.ylabel('Delta Pitch')
            fig.colorbar(im, ax=axes.ravel().tolist())

class VisTraining(PlotFunc):

    @staticmethod
    def description():
        return 'visualize training dataset'

    def _plot(self, data):
        n_q_perc_bins = 6
        n_limit_bins = 4
        q_percs = np.linspace(0, 1.2, n_q_perc_bins+1)
        limits = np.linspace(0.05, 0.25, n_limit_bins+1)

        plot_data = {}
        for point in data:
            limit = point.mechanism_params.params.range/2
            closest_lim = min(limits, key=lambda x: abs(x-limit))
            closest_lim_i = list(limits).index(closest_lim)
            if closest_lim > limit or closest_lim_i == n_limit_bins:
                closest_lim_i -= 1

            q_perc = abs(point.config_goal/limit)
            closest_q_perc = min(q_percs, key=lambda x: abs(x-q_perc))
            closest_q_perc_i = list(q_percs).index(closest_q_perc)
            if closest_q_perc > q_perc or closest_q_perc_i == n_q_perc_bins:
                closest_q_perc_i -= 1

            if (closest_lim_i, closest_q_perc_i) not in plot_data:
                plot_data[(closest_lim_i, closest_q_perc_i)] = [[point.policy_params.delta_values.delta_yaw],
                                                                [point.policy_params.delta_values.delta_pitch],
                                                                [point.net_motion]]
            else:
                plot_data[(closest_lim_i, closest_q_perc_i)][0] += [point.policy_params.delta_values.delta_yaw]
                plot_data[(closest_lim_i, closest_q_perc_i)][1] += [point.policy_params.delta_values.delta_pitch]
                plot_data[(closest_lim_i, closest_q_perc_i)][2] += [point.net_motion]
        fig, axes = plt.subplots(n_limit_bins, n_q_perc_bins)
        for i in range(n_limit_bins):
            for j in range(n_q_perc_bins):
                if (i,j) in plot_data:
                    minm = min(plot_data[i,j][2])
                    maxm = max(plot_data[i,j][2])
                    im = axes[i,j].scatter(plot_data[i,j][0], plot_data[i,j][1], c=plot_data[i,j][2], vmin=minm, vmax=maxm)
                    if j==0:
                        limit_min = str(round(limits[i],2))
                        limit_max = str(round(limits[i+1],2))
                        axes[i,j].set_ylabel('limit=['+limit_min+','+limit_max+']')
                    if i==0:
                        q_perc_min = str(round(q_percs[j],2))
                        q_perc_max = str(round(q_percs[j+1],2))
                        axes[i,j].set_title('q%=['+q_perc_min+','+q_perc_max+']')
                    fig.colorbar(im, ax=axes[i,j])

## PLOTS THAT USE A MODEL FILE ##

class YawPitchMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a heatmap of the predicted motion for yaw versus pitch for several q values'

    def _plot(self, model):
        n_policy_samples = 200
        n_configs = 9
        n_mechs = 5
        goal_configs_perc = np.linspace(-1.2, 1.2, n_configs)
        goal_configs = np.zeros((n_mechs, n_configs))
        true_yaws = np.zeros((n_mechs, n_configs, n_policy_samples))
        true_pitches = np.zeros((n_mechs, n_configs, n_policy_samples))
        true_motions = np.zeros((n_mechs, n_configs, n_policy_samples))
        pred_yaws = np.zeros((n_mechs, n_configs, n_policy_samples))
        pred_pitches = np.zeros((n_mechs, n_configs, n_policy_samples))
        pred_motions = np.zeros((n_mechs, n_configs, n_policy_samples))
        mech_limits = []
        for m in range(n_mechs):
            bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider], urdf_tag='plot')
            mech_limits += [bb._mechanisms[0].range/2]
            for (j, goal_config_perc) in enumerate(goal_configs_perc):
                # get ground truth motion for random policies
                results = []
                for _ in range(n_policy_samples):
                    result = generate_samples(False, False, 1, True, 1.0, goal_config_perc, bb)[0]
                    results += [result]
                true_yaws[m,j,:] = [result.policy_params.delta_values.delta_yaw for result in results]
                true_pitches[m,j,:] = [result.policy_params.delta_values.delta_pitch for result in results]
                true_motions[m,j,:] = [result.net_motion for result in results]
                goal_configs[m,j] = result.config_goal
                # get predicted motion for same policies
                data = parse_pickle_file(data=results)
                dataset = PolicyDataset(data)
                for i in range(len(dataset.items)):
                    policy_type = dataset.items[i]['type']
                    policy_params = dataset.tensors[i].unsqueeze(0)
                    pred_motions[m,j,i] = model.forward(policy_type,
                                                policy_params,
                                                dataset.configs[i].unsqueeze(0),
                                                dataset.images[i].unsqueeze(0))
                    policy = get_policy_from_params(policy_type, policy_params[0].numpy())
                    pred_yaws[m,j,i] = dataset.delta_vals[i].delta_yaw
                    pred_pitches[m,j,i] = dataset.delta_vals[i].delta_pitch

        lw = int(round(np.sqrt(n_configs)))
        for m in range(n_mechs):
            min_motion = min(min(true_motions[m,:,:].flatten()), min(pred_motions[m,:,:].flatten()))
            max_motion = max(max(true_motions[m,:,:].flatten()), max(pred_motions[m,:,:].flatten()))
            fig, axes = plt.subplots(lw, 2*lw, sharex=True, sharey=True)
            plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
            ax_left = axes[:,:lw]
            ax_right = axes[:,lw:]
            config_num = 0
            for ax in ax_left.flatten():
                if config_num < n_configs:
                    im = ax.scatter(true_yaws[m,config_num,:],
                                    true_pitches[m,config_num,:],
                                    c=true_motions[m,config_num,:],
                                    vmin=min_motion, vmax=max_motion)
                    ax.set_title('q='+str(round(goal_configs[m,config_num],2))+', l='+str(round(mech_limits[m],2)))
                    config_num += 1
            config_num = 0
            for ax in ax_right.flatten():
                if config_num < n_configs:
                    im = ax.scatter(pred_yaws[m,config_num,:],
                                    pred_pitches[m,config_num,:],
                                    c=pred_motions[m,config_num,:],
                                    vmin=min_motion, vmax=max_motion)
                    ax.set_title('q='+str(round(goal_configs[m,config_num],2))+', l='+str(round(mech_limits[m],2)))
                    config_num += 1
            im.set_clim(min_motion, max_motion)
            fig.colorbar(im, ax=axes.ravel().tolist())

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
    if file_name[-3:] == '.pt':
        data = util.load_model(file_name)
    else:
        data = util.read_from_file(file_name)
    try:
        if type(data[0]) == util.Result:
            print_stats(data)
    except:
        pass

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

    # TODO: give each plt a figure number then use those to save multiple plots
    '''
    save_plots = input('save plots? [y/n]')
    if save_plots == 'y':
        plt.savefig('plots.png', bbox_inches='tight')
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fname', type=str, required=True) # give filename (without .pickle)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_results(args.fname)
