import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle

if __name__ == '__main__':

    c_lookup = {
        'GP-UCB': 'g',
        'Active-NN GP-UCB': 'k',
        'GP-UCB-NN GP-UCB': 'r',
        'Systematic': 'b',
        'Random-NN GP-UCB': 'y'
    }

    gpucb_fnames = {
        0: 'regret_results_gpucb_t0_n20.pickle',
        2: 'regret_results_gpucb_t2_n20.pickle',
        5: 'regret_results_gpucb_t5_n20.pickle',
        10: 'regret_results_gpucb_t10_n20.pickle',
    }

    active_fnames = {
        0: 'regret_results_active_nn_t0_n20.pickle',
        2: 'regret_results_active_nn_t2_n20.pickle',
        5: 'regret_results_active_nn_t5_n20.pickle',
    }

    gpucb_nn_fnames = {
        0: 'regret_results_gpucb_nn_t0_n20.pickle',
        2: 'regret_results_gpucb_nn_t2_n20.pickle',
        5: 'regret_results_gpucb_nn_t5_n20.pickle',
    }

    random_nn_fnames = {
        2: 'regret_results_random_nn_t2_n20.pickle',
    }

    systematic_fnames = {
        2: 'systematic_n20_t2.pickle',
        5: 'systematic_n20_t5.pickle',
        10: 'systematic_n20_t10.pickle'
    }



    for n_interactions in [0, 2, 5, 10]:

        for name, result_lookup in zip(['GP-UCB', 'Active-NN GP-UCB', 'GP-UCB-NN GP-UCB', 'Systematic', 'Random-NN GP-UCB'],
                                       [gpucb_fnames, active_fnames, gpucb_nn_fnames, systematic_fnames, random_nn_fnames]):
            if n_interactions not in result_lookup:
                continue

            xs = range(10, 101, 10)
            with open(result_lookup[n_interactions], 'rb') as handle:
                results = pickle.load(handle)

            if name == 'GP-UCB':
                rs = [results[0]['final']] * 10
                s = [np.std(results[0]['regrets'])] * 10
                med = [np.median(results[0]['regrets'])] * 10
                q25 = [np.quantile(results[0]['regrets'], 0.25)] * 10
                q75 = [np.quantile(results[0]['regrets'], 0.75)] * 10
            elif name == 'Systematic':
                rs = [results['final']] * 10
                s = [np.std(results['min_regrets'])] * 10
                med = [np.median(results['min_regrets'])] * 10
                q25 = [np.quantile(results['min_regrets'], 0.25)] * 10
                q75 = [np.quantile(results['min_regrets'], 0.75)] * 10
            elif name == 'Random-NN GP-UCB':
                # Temporarily remove 500 and 1500
                del results[2]
                del results[0]
                rs = [res['final'] for res in results]
                med = [np.median(res['regrets']) for res in results]
                s = [np.std(res['regrets']) for res in results]
                q25 = [np.quantile(res['regrets'], 0.25) for res in results]
                q75 = [np.quantile(res['regrets'], 0.75) for res in results]
            else:
                rs = [res['final'] for res in results]
                med = [np.median(res['regrets']) for res in results]
                s = [np.std(res['regrets']) for res in results]
                q25 = [np.quantile(res['regrets'], 0.25) for res in results]
                q75 = [np.quantile(res['regrets'], 0.75) for res in results]

            print(q25, q75)
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


