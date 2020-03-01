import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
import argparse
import os
from utils import util
import itertools
import re

def get_success(regrets, std=False):
    success = []
    for r in regrets:
        if r < 0.05:
            success.append(1)
        else:
            success.append(0)

    p = np.mean(success)
    p_std = np.sqrt(p*(1-p)/len(success))

    return p, p_std

def get_result_file(type_name, results_path, noT):
    all_files = os.walk(results_path)
    result_files = {}
    for root, subdir, files in all_files:
        for file in files:
            if 'regret_results_' in file:
                if type_name in file:
                    if not noT:
                        T_result = re.search('regret_results_(.*)_(.*)T_(.*)N_(.*)M.pickle', file)
                        T, N = T_result.group(2,3)
                        result_files[(T, N)] = root + '/' + file
                    elif noT:
                        results = re.search('regret_results_noT_(.*)_(.*)N_(.*).pickle', file)
                        N = results.group(2)
                        if not N in result_files:
                            result_files[N] = [root + '/' + file]
                        else:
                            result_files[N] += [root + '/' + file]
    return result_files

def make_regret_T_plots(res_files):
    for (T,N), res_file in res_files.items():
        if T in plt_axes:
            pass
        else:
            _, median_ax = plt.subplots()
            _, mean_ax = plt.subplots()
            _, succ_ax = plt.subplots()
            plt_axes[T] = [median_ax, mean_ax, succ_ax]
        regret_results = util.read_from_file(res_file, verbose=False)
        Ls = sorted(regret_results.keys())

        mean_regrets = []
        median_regrets = []
        std_dev_regrets = []
        q25_regrets = []
        q75_regrets = []
        prob_successes = []
        std_successes = []
        for L in Ls:
            all_L_regrets = list(itertools.chain.from_iterable([regret_results[L][model] \
                                for model in regret_results[L].keys()]))
            mean_regrets += [np.mean(all_L_regrets)]
            median_regrets += [np.median(all_L_regrets)]
            std_dev_regrets += [np.std(all_L_regrets)]
            q25_regrets += [np.quantile(all_L_regrets, 0.25)]
            q75_regrets += [np.quantile(all_L_regrets, 0.75)]
            prob_success, std_success = get_success(all_L_regrets)
            prob_successes += [prob_success]
            std_successes += [std_success]

        med_bot, med_mid, med_top = q25_regrets, median_regrets, q75_regrets  # Quantiles
        mean_bot, mean_mid, mean_top = np.subtract(mean_regrets, std_dev_regrets), \
                        mean_regrets, \
                        np.add(mean_regrets, std_dev_regrets)  # Standard Deviation
        succ_bot, succ_mid, succ_top = np.subtract(prob_successes, std_successes), \
                        prob_successes, \
                        np.add(prob_successes, std_successes)  # Success

        for (bot, mid, top, type, ax) in ((med_bot, med_mid, med_top, 'Median Regret', plt_axes[T][0]),\
                                (mean_bot, mean_mid, mean_top, 'Mean Regret', plt_axes[T][1]),\
                                (succ_bot, succ_mid, succ_top, '% Success', plt_axes[T][2])):
            #plt.figure()
            for plot_type in plot_info:
                if plot_type in name:
                    plot_params = plot_info[plot_type]
            ax.plot(Ls, mid, c=plot_params[0], label=plot_params[1])
            ax.fill_between(Ls, bot, top, facecolor=plot_params[0], alpha=0.2)

            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_xlabel('L')
            ax.set_ylabel(type)
            ax.set_title('Evaluated on T=%s Interactions on N=%s Mechanisms' % (T, N))

def add_baseline_to_ax(fname, type, mean_ax, median_ax):
    plot_params = plot_info[type]
    steps = util.read_from_file(fname)
    mean = [np.mean(steps)]*10
    std = [np.std(steps)]*10
    median = [np.median(steps)]*10
    q25 = [np.quantile(steps, 0.25)]*10
    q75 = [np.quantile(steps, 0.75)]*10
    Ls = list(range(10,101,10))
    mean_ax.plot(Ls, mean, c=plot_params[0], label=plot_params[1])
    mean_ax.fill_between(Ls, np.subtract(mean, std), np.add(mean,std), facecolor=plot_params[0], alpha=0.2)
    median_ax.plot(Ls, median, c=plot_params[0], label=plot_params[1])
    median_ax.fill_between(Ls, q25, q75, facecolor=plot_params[0], alpha=0.2)

def make_regret_noT_plots(types, res_path):
    _, median_ax = plt.subplots()
    _, mean_ax = plt.subplots()

    add_baseline_to_ax(args.results_path+'/random_sliders_50N.pickle', 'systematic', mean_ax, median_ax)
    add_baseline_to_ax(args.results_path+'/gpucb_sliders_50N.pickle', 'gpucb_baseline', mean_ax, median_ax)

    N_plot = None
    for name in types:
        res_files = get_result_file(name, res_path, True)
        print(res_files)
        for N, res_files in res_files.items():
            print(N)
            if N_plot is None:
                N_plot = N
            assert N_plot == N, \
                    'You are trying to plot results with different N values. Please \
check that all results on the results path have the same N value'
            all_L_steps = {}
            for res_file in res_files:
                regret_results = util.read_from_file(res_file, verbose=False)
                Ls = sorted(regret_results.keys())
                for L in Ls:
                    L_steps = list(itertools.chain.from_iterable([regret_results[L][model] \
                                        for model in regret_results[L].keys()]))
                    if L not in all_L_steps:
                        all_L_steps[L] = L_steps
                    else:
                        all_L_steps[L] += L_steps
            Ls = sorted(all_L_steps.keys())
            mean_steps = [np.mean(all_L_steps[L]) for L in Ls]
            std_dev_steps = [np.std(all_L_steps[L]) for L in Ls]

            median_steps = [np.median(all_L_steps[L]) for L in Ls]
            q25_steps = [np.quantile(all_L_steps[L], 0.25) for L in Ls]
            q75_steps = [np.quantile(all_L_steps[L], 0.75) for L in Ls]

            med_bot, med_mid, med_top = q25_steps, median_steps, q75_steps  # Quantiles
            mean_bot, mean_mid, mean_top = np.subtract(mean_steps, std_dev_steps), \
                            mean_steps, \
                            np.add(mean_steps, std_dev_steps)  # Standard Deviation

            for (bot, mid, top, type, ax) in ((med_bot, med_mid, med_top, 'Median Interactions', mean_ax),\
                                    (mean_bot, mean_mid, mean_top, 'Mean Interactions', median_ax)):
                for plot_type in plot_info:
                    if plot_type in name:
                        plot_params = plot_info[plot_type]
                ax.plot(Ls, mid, c=plot_params[0], label=plot_params[1])
                ax.fill_between(Ls, bot, top, facecolor=plot_params[0], alpha=0.2)

                ax.set_ylim(top=110, bottom=0)
                ax.set_xlim(left=10.0, right=100.0)
                ax.legend()
                ax.set_xlabel('L')
                ax.set_ylabel(type)
                ax.set_title('Interactions to Success on N=%s Mechanisms' % N)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', nargs='+', help='list types of datasets to plot, pick from :[active, gpucb, systematic, random]')
    parser.add_argument('--results-path', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--noT', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_info = {
        'active': ['k', 'Train-Active'],
        'gpucb': ['r', 'Train-GP-UCB'],
        'systematic': ['b', 'Random'],
        'random': ['c', 'Train-Random'],
        'gpucb_baseline': ['g', 'GP-UCB']
    }
    plt.ion()
    #plt_axes = {}

    if not args.noT:
        make_regret_T_plots(res_files)
    elif args.noT:
        make_regret_noT_plots(args.types, args.results_path)

    plt.show()
    input('enter to close')
    plt.close()
