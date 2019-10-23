import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
import argparse
import os
from utils import util
import itertools

def get_success(regrets, std=False):
    success = []
    for r in regrets:
        if r < 0.05:
            success.append(1)
        else:
            success.append(0)

    p = np.mean(success)
    p_std = np.sqrt(p*(1-p)/len(success))

    if std:
        return p_std
    else:
        return p

def get_result_file(type_name, T, results_path):
    all_files = os.walk(results_path)
    for root, subdir, files in all_files:
        for file in files:
            if ('regret_results' in file) and (type_name in file) and (str(T)+'T_' in file):
                return root + '/' + file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', nargs='+', help='list types of datasets to plot, pick from :[active, gpucb, systematic, random]')
    parser.add_argument('--T', type=int)
    parser.add_argument('--results-path', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_info = {
        'active': ['k', 'Train-Active'],
        'gpucb': ['r', 'Train-GP-UCB'],
        'systematic': ['b', 'Systematic'],
        'random': ['c', 'Train-Random']
    }

    print('T=', args.T)

    for name in args.types:
        res_file = get_result_file(name, args.T, args.results_path)
        regret_results = util.read_from_file(res_file)
        Ls = regret_results.keys()

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
            prob_successes += [get_success(all_L_regrets)]
            std_successes += [get_success(all_L_regrets, std=True)]

        bot, mid, top = q25_regrets, median_regrets, q75_regrets  # Quantiles
        # bot, mid, top = rs - s, rs, rs + s  # Standard Deviation
        # bot, mid, top = p - p_std, p, p + p_std  # Success
        
        plt.plot(Ls, mid, c=plot_info[name][0], label=plot_info[name][1])
        plt.fill_between(Ls, bot, top, facecolor=plot_info[name][0], alpha=0.2)


        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('L')
        plt.ylabel('Regret')

        plt.show()
