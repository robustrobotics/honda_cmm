import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle

if __name__ == '__main__':

    nn_regrets = [0.5696455953137332,
                  0.46496730194840374,
                  0.29263590867999206,
                  0.37526911291086296,
                  0.29902841288323817]
    # csv_fname = 'run-.-tag-Test_Regret_Average_Regret.csv'
    # with open(csv_fname, 'r') as handle:
    #     reader = csv.reader(handle)
    #     for row in reader:
    #         if len(row) > 1:
    #             nn_regrets.append(row[-1])
    #     nn_regrets = nn_regrets[1:]
    nn_regrets = [float(x) for x in nn_regrets]

    gp_avg_5 = [0.56] * len(nn_regrets)
    gp_avg_10 = [0.50] * len(nn_regrets)
    gp_avg_20 = [0.57] * len(nn_regrets)

    gp_final_5 = [0.24] * len(nn_regrets)
    gp_final_10 = [0.2] * len(nn_regrets)
    gp_final_20 = [0.06] * len(nn_regrets)

    with open('regret_results_conv2_2.pickle', 'rb') as handle:
        data = pickle.load(handle)
        xs_short = []
        ys_nn_avg = []
        ys_nn_final = []
        for ix, entry in enumerate(data):
            if ix == 0:
                gp_final_10 = [entry['final']]
                gp_avg_10 = [entry['avg']]
            else:
                xs_short.append(10*(ix))
                ys_nn_avg.append(entry['avg'])
                ys_nn_final.append(entry['final'])

    # gp_final_10 = [0.48] * len(xs_short)

    gp_avg_10 = gp_avg_10 * len(xs_short)
    gp_final_10 = gp_final_10 * len(xs_short)

    xs = np.arange(1, len(nn_regrets)+1)*10

    # plt.plot(xs, nn_regrets, c='r', label='NN')
    # plt.plot(xs, gp_final_5, c='b', label='GP-5')
    plt.plot(xs_short, gp_final_10, c='g', label='GP-2')
    # plt.plot(xs, gp_final_20, c='y', label='GP-20')
    plt.plot(xs_short, ys_nn_final, c='k', label='NN-GP-2')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Final Regret')
    plt.xlabel('# BB')
    plt.ylabel('Regret')
    plt.show()


