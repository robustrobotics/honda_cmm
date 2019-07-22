import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from util import util
import sys
from learning.test_model import SearchResult, SampleResult, get_pred_motions
from actions.policies import PrismaticParams, RevoluteParams, get_policy_from_params
from gen.generator_busybox import BusyBox, Slider, Door
from gen.generate_policy_data import generate_samples
from learning.dataloaders import parse_pickle_file, PolicyDataset, create_data_splits
import actions.policies as policies

class PlotFunc(object):

    @staticmethod
    def description():
        return 'No description for PlotFunc: ' + self.__name__

class DoorRadiusMotion(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the door radius versus the joint motion'

    def _plot(self, data, model=None):
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

    def _plot(self, data, model=None):
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

    def _plot(self, data, model=None):
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

class MotionKD(PlotFunc):

    @staticmethod
    def description():
        return 'Plot a heatmap of the motion reached for varying k and d values (fixed q)'

    def _plot(self, data, model=None):
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

    def _plot(self, data, model=None):
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

    def _plot(self, data, model=None):
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
        return 'Plot the motion generated versus to distance from the true slider policy for all goal configurations (no z axis)'

    def _plot(self, data, model=None):
        fig, ax = plt.subplots()
        delta_yaws = [data_point.policy_params.delta_values.delta_yaw for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        delta_pitches = [data_point.policy_params.delta_values.delta_pitch for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        motion = [data_point.net_motion for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        im = ax.scatter(delta_yaws, delta_pitches, c=motion, vmin=min(motion), vmax=max(motion))
        fig.colorbar(im)
        ax.set_xlabel('Delta Yaw')
        ax.set_ylabel('Delta Pitch')
        ax.set_title('Motion for Varying Prismatic Policy Values')

class SliderPolicyDeltaConfig(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the motion generated versus to distance from the true slider policy and the goal configuration (along the z axis)'

    def _plot(self, data, model=None):
        from mpl_toolkits.mplot3d import axes3d, Axes3D

        fig = plt.figure()
        ax = Axes3D(fig)
        delta_yaws = [data_point.policy_params.delta_values.delta_yaw for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        delta_pitches = [data_point.policy_params.delta_values.delta_pitch for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        goal_configs =  [data_point.config_goal for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        motion = [data_point.net_motion for data_point in plot_data \
                        if data_point.pose_joint_world_final is not None]
        im = ax.scatter(delta_yaws, delta_pitches, goal_configs, c=motion, vmin=min(motion), vmax=max(motion))
        fig.colorbar(im)
        ax.set_xlabel('Delta Yaw')
        ax.set_ylabel('Delta Pitch')
        ax.set_title('Motion for Varying Prismatic Policy Values')

class DoorPolicyDelta(PlotFunc):

    @staticmethod
    def description():
        return 'Plot the motion generated versus to distance from the true door policy'

    def _plot(self, data, model=None):
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

    def _plot(self, data, model=None):
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

class VisTrainingPerformance(PlotFunc):

    @staticmethod
    def description():
        return 'generate plots to visualize training performance'

    def _plot(self, data, model=None):
        train_data, val_data, test_data = create_data_splits(data)
        self._plot_data(train_data, 'Training Data')
        self._plot_data(val_data, 'Validation Prediction Error', model)
        self._plot_data(test_data, 'Test Prediction Error', model)

    def _plot_data(self, data, title, model=None):
        n_q_perc_bins = 6
        n_limit_bins = 4
        q_percs = np.linspace(0, 1.2, n_q_perc_bins+1)
        limits = np.linspace(0.05, 0.25, n_limit_bins+1)
        plot_data = {}
        for point in data:
            if model is not None:
                pred_motion = get_pred_motions([point], model)[0]
                plot_motion = abs(point.net_motion - pred_motion)
            else:
                plot_motion = point.net_motion
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
                                                                [plot_motion]]
            else:
                plot_data[(closest_lim_i, closest_q_perc_i)][0] += [point.policy_params.delta_values.delta_yaw]
                plot_data[(closest_lim_i, closest_q_perc_i)][1] += [point.policy_params.delta_values.delta_pitch]
                plot_data[(closest_lim_i, closest_q_perc_i)][2] += [plot_motion]
        if model is not None:
            minm = min([min(plot_data[i,j][2]) for i in range(n_limit_bins) \
                                                for j in range(n_q_perc_bins)])
            maxm = max([max(plot_data[i,j][2]) for i in range(n_limit_bins) \
                                                for j in range(n_q_perc_bins)])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        axes = fig.subplots(n_limit_bins, n_q_perc_bins)
        for i in range(n_limit_bins):
            for j in range(n_q_perc_bins):
                if (i,j) in plot_data:
                    if model is None:
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
                    if model is None:
                        fig.colorbar(im, ax=axes[i,j])
        if model is not None:
            fig.colorbar(im, ax=axes.ravel().tolist())

class TestMechs(PlotFunc):

    @staticmethod
    def description():
        return 'show model performance on test mechanisms'

    def _plot(self, data, model):
        n_mechs = 9
        n_samples = 500
        delta_yaws = np.zeros((n_mechs, n_samples))
        delta_pitches = np.zeros((n_mechs, n_samples))
        motions = np.zeros((n_mechs, n_samples))
        limits = []
        for i in range(n_mechs):
            bb = BusyBox.generate_random_busybox(max_mech=1, mech_types=[Slider])
            image_data = util.setup_pybullet.setup_env(bb, False, False)
            mech = bb._mechanisms[0]
            limits += [mech.range/2.0]
            mech_tuple = mech.get_mechanism_tuple()
            for j in range(n_samples):
                random_policy = policies.generate_policy(bb, mech, True, 1.0)
                policy_type = random_policy.type
                policy_tuple = random_policy.get_policy_tuple()
                delta_yaws[i,j] = policy_tuple.delta_values.delta_yaw
                delta_pitches[i,j] = policy_tuple.delta_values.delta_pitch
                sample = util.Result(policy_tuple, mech_tuple, 0.0, 0.0,
                                        None, None, limits[i], image_data, None, 1.0)
                pred_motion = get_pred_motions([sample], model)
                motions[i,j] = pred_motion[0].detach().numpy()

        lw = int(round(np.sqrt(n_mechs)))
        fig, axes = plt.subplots(lw, lw, sharex=True, sharey=True)
        plt.setp(axes.flat, aspect=1.0, adjustable='box-forced')
        for (n, ax) in enumerate(axes.flatten()):
            if n < n_mechs:
                min_motion = min(motions[n,:])
                max_motion = max(motions[n,:])
                im = ax.scatter(delta_yaws[n,:],
                                delta_pitches[n,:],
                                c=motions[n,:],
                                vmin=min_motion, vmax=max_motion)
                ax.set_title('limit = '+str(round(limits[n], 2)))
                fig.colorbar(im, ax=ax)
                ax.set_xlabel('Delta Yaw')
                ax.set_ylabel('Delta Pitch')

class ValError(PlotFunc):

    @staticmethod
    def description():
        return 'plot validation prediction error scatterplot'

    def _plot(self, data, model):
        train_data, val_data, test_data = create_data_splits(data)
        self._plot_data(val_data, model)

    def _plot_data(self, data, model):
        from learning import viz
        loss_fn = torch.nn.MSELoss()
        val_losses = []
        ys, yhats, types = [], [], []
        for point in data:
            yhat = get_pred_motions([point], model)[0]
            y = point.net_motion
            loss = loss_fn(yhat, y)
            val_losses.append(loss.item())
            types += [point.policy_params.type]
            ys += y.numpy().tolist()
            yhats += yhat.detach().numpy().tolist()
        viz.plot_y_yhat(ys, yhats, types, ex, title='PolVis')

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

def plot_results(file_name, model):
    data = None
    if file_name:
        data = util.read_from_file(file_name)
        print_stats(data)
    if model is not None:
        model = util.load_model(model)

    plot_funcs = PlotFunc.__subclasses__()
    for (i, func) in enumerate(plot_funcs):
        print(i, ':', func.description())
    plot_nums = input('Above are the possible plots, type in the numbers of the plots you would like to visualize, eg. 1, 3, 4 [ENTER]\n')
    plot_nums = list(map(int, plot_nums.strip('[]').split(',')))

    plt.ion()
    for plot_num in plot_nums:
        plot_func = plot_funcs[plot_num]()
        plot_func._plot(data, model)
    plt.show()
    input('hit [ENTER] to close plots')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_results(args.fname, args.model)
