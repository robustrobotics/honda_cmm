import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == '__main__':
    nn_regrets = []
    csv_fname = 'run-.-tag-Test_Regret_Average_Regret.csv'
    with open(csv_fname, 'r') as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) > 1:
                nn_regrets.append(row[-1])
        nn_regrets = nn_regrets[1:]
    nn_regrets = [float(x) for x in nn_regrets]

    gp_avg_5 = [0.5262] * len(nn_regrets)
    gp_avg_10 = [0.5522] * len(nn_regrets)
    gp_avg_20 = [0.5848] * len(nn_regrets)

    gp_final_5 = [0.5847] * len(nn_regrets)
    gp_final_10 = [0.118] * len(nn_regrets)
    gp_final_20 = [0] * len(nn_regrets)

    xs = np.arange(0, len(nn_regrets))
    print(nn_regrets)
    plt.plot(xs, nn_regrets, c='r', label='NN')
    plt.plot(xs, gp_avg_5, c='b', label='GP-5')
    plt.plot(xs, gp_avg_10, c='g', label='GP-10')
    plt.plot(xs, gp_avg_20, c='y', label='GP-20')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Avg. Regret')
    plt.xlabel('# BB')
    plt.ylabel('Regret')
    plt.show()

    plt.plot(xs, nn_regrets, c='r', label='NN')
    plt.plot(xs, gp_final_5, c='b', label='GP-5')
    plt.plot(xs, gp_final_10, c='g', label='GP-10')
    plt.plot(xs, gp_final_20, c='y', label='GP-20')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('Final Regret')
    plt.xlabel('# BB')
    plt.ylabel('Regret')
    plt.show()


