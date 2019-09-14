import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['train_eval', 'test_eval'])
    args = parser.parse_args()

    c_lookup = {
        'GP-UCB': 'g',
        'Active-NN GP-UCB': 'k',
        'GP-UCB-NN GP-UCB': 'r',
        'Systematic': 'b',
        'Random-NN GP-UCB': 'y'
    }

    gpucb_fnames = {
        0: ['regrets/regret_results_gpucb_t0_n50.pickle'],
        2: ['regrets/regret_results_gpucb_t2_n50.pickle'],
        5: ['regrets/regret_results_gpucb_t5_n50.pickle'],
        10: ['regrets/regret_results_gpucb_t10_n50.pickle'],
    }

    active_fnames = {
        0: ['regrets/regret_results_active_nn_t0_n50.pickle',
            'regrets/regret_results_active_nn_t0_n50_2.pickle'],
        2: ['regrets/regret_results_active_nn_t2_n20.pickle'],
        5: ['regrets/regret_results_active_nn_t5_n20.pickle'],
        10: ['regrets/regret_results_active_nn_t10_n20.pickle'],
    }

    gpucb_nn_fnames = {
        0: ['regrets/regret_results_gpucb_nn_t0_n50.pickle',
            'regrets/regret_results_gpucb_nn_t0_n50_2.pickle',
            'regrets/regret_results_gpucb_nn_t0_n50_3.pickle'],
        2: ['regrets/regret_results_gpucb_nn_t2_n50.pickle',
            'regrets/regret_results_gpucb_nn_t2_n50_2.pickle',
            'regrets/regret_results_gpucb_nn_t2_n50_3.pickle'],
        5: ['regrets/regret_results_gpucb_nn_t5_n20.pickle',
            'regrets/regret_results_gpucb_nn_t5_n50_2.pickle',
            'regrets/regret_results_gpucb_nn_t5_n50_3.pickle'],
        10: ['regrets/regret_results_gpucb_nn_t10_n50_2.pickle',
                'regrets/regret_results_gpucb_nn_t10_n50_3.pickle'],

    }

    random_nn_fnames = {
        0: ['regrets/regret_results_random_nn_t0_n50_run1.pickle',
            'regrets/regret_results_random_nn_t0_n50_run2.pickle',
            'regrets/regret_results_random_nn_t0_n50_run3.pickle'],
        2: ['regrets/regret_results_random_nn_t2_n50_run1.pickle',
            'regrets/regret_results_random_nn_t2_n50_run2.pickle',
            'regrets/regret_results_random_nn_t2_n50_run3.pickle'],
        5: ['regrets/regret_results_random_nn_t5_n50_run1.pickle',
            'regrets/regret_results_random_nn_t5_n50_run2.pickle',
            'regrets/regret_results_random_nn_t5_n50_run3.pickle'],
        10: ['regrets/regret_results_random_nn_t10_n50_run1.pickle',
            'regrets/regret_results_random_nn_t0_n50_run2.pickle',
            'regrets/regret_results_random_nn_t0_n50_run3.pickle'],
    }

    systematic_fnames = {
        2: ['regrets/systematic_n20_t2.pickle'],
        5: ['regrets/systematic_n20_t5.pickle'],
        10: ['regrets/systematic_n20_t10.pickle']
    }


    for n_interactions in [0, 2, 5, 10]:

        for name, result_lookup in zip(['GP-UCB', 'Active-NN GP-UCB', 'GP-UCB-NN GP-UCB', 'Systematic', 'Random-NN GP-UCB'],
                                       [gpucb_fnames, active_fnames, gpucb_nn_fnames, systematic_fnames, random_nn_fnames]):
            if n_interactions not in result_lookup:
                continue

            if (name == 'GP-UCB' or name == 'Systematic' or name == 'Active-NN GP-UCB') and args.type == 'train_eval':
                continue
            elif (name == 'Active-NN GP-UCB' or name == 'Random-NN GP-UCB') and args.type == 'test_eval':
                continue

            xs = range(10, 101, 10)
            results = None
            for res_file in result_lookup[n_interactions]:
                with open(res_file, 'rb') as handle:
                    cur_results = pickle.load(handle)

                    if results is None:
                        results = cur_results
                    else:
                        for ix in range(len(results)):
                            results[ix]['regrets'].extend(cur_results[ix]['regrets'])

            if name == 'GP-UCB':
                rs = [np.mean(results[0]['regrets'])] * 10
                s = [np.std(results[0]['regrets'])] * 10
                med = [np.median(results[0]['regrets'])] * 10
                q25 = [np.quantile(results[0]['regrets'], 0.25)] * 10
                q75 = [np.quantile(results[0]['regrets'], 0.75)] * 10
            elif name == 'Systematic':
                rs = [np.mean(results['min_regrets'])] * 10
                s = [np.std(results['min_regrets'])] * 10
                med = [np.median(results['min_regrets'])] * 10
                q25 = [np.quantile(results['min_regrets'], 0.25)] * 10
                q75 = [np.quantile(results['min_regrets'], 0.75)] * 10
            elif name == 'Random-NN GP-UCB':
                # Temporarily remove 500 and 1500
                del results[2]
                del results[0]
                rs = [np.mean(res['regrets']) for res in results]
                med = [np.median(res['regrets']) for res in results]
                s = [np.std(res['regrets']) for res in results]
                q25 = [np.quantile(res['regrets'], 0.25) for res in results]
                q75 = [np.quantile(res['regrets'], 0.75) for res in results]
            else:
                rs = [np.mean(res['final']) for res in results]
                med = [np.median(res['regrets']) for res in results]
                s = [np.std(res['regrets']) for res in results]
                q25 = [np.quantile(res['regrets'], 0.25) for res in results]
                q75 = [np.quantile(res['regrets'], 0.75) for res in results]

            rs, s = np.array(rs), np.array(s)

            bot, mid, top = q25, med, q75  # Quantiles
            # bot, mid, top = rs - s, rs, rs + s  # Standard Deviation

            plt.plot(xs, mid, c=c_lookup[name], label=name)

            plt.fill_between(xs, bot, top, facecolor=c_lookup[name], alpha=0.2)

        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('# BB')
        plt.ylabel('Regret')
        plt.show()
