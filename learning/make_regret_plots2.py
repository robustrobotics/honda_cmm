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

def get_result_file(type_name, results_path):
    all_files = os.walk(results_path)
    result_files = {}
    for root, subdir, files in all_files:
        for file in files:
            if ('regret_results' in file) and (type_name in file):
                T_result = re.search('regret_results_(.*)_(.*)T_(.*)N_(.*)M.pickle', file)
                T, N = T_result.group(2,3)
                result_files[(T, N)] = root + '/' + file
    return result_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--types', nargs='+', help='list types of datasets to plot, pick from :[active, gpucb, systematic, random]')
    parser.add_argument('--results-path', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    plot_info = {
        'active': ['k', 'Train-Active'],
        'gpucb': ['r', 'Train-GP-UCB'],
        'systematic': ['b', 'Systematic'],
        'doors_random': ['c', 'Train-Random']
    }
    plt.ion()
    for name in args.types:
        res_files = get_result_file(name, args.results_path)
        for (T,N), res_file in res_files.items():
            regret_results = util.read_from_file(res_file)
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

            for (bot, mid, top, type) in ((med_bot, med_mid, med_top, 'Median Regret'),\
                                    (mean_bot, mean_mid, mean_top, 'Mean Regret'),\
                                    (succ_bot, succ_mid, succ_top, '% Success')):
                plt.figure()
                plt.plot(Ls, mid, c=plot_info[name][0], label=plot_info[name][1])
                plt.fill_between(Ls, bot, top, facecolor=plot_info[name][0], alpha=0.2)

                plt.ylim(0, 1)
                plt.legend()
                plt.xlabel('L')
                plt.ylabel(type)
                plt.title('Evaluated on T=%s Interactions on N=%s Mechanisms' % (T, N))

            plt.show()
            input('enter to close')
            plt.close()
