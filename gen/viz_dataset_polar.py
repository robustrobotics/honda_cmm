import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from gen.generator_busybox import BusyBox

def plot_result(result, ax):
    cmap = plt.get_cmap('viridis')
    c = cmap(result.net_motion/(result.mechanism_params.params.range/2.0))
    x = [result.policy_params.params.pitch, result.config_goal]
    if x[1] < 0:
        ax.scatter(x[0] - np.pi, -1 * x[1], c=c, s=10)
    else:
        ax.scatter(x[0], x[1], c=c, s=10)

def viz_circles(results):
    plt.figure(figsize=(20, 5))

    # Plot the BusyBox
    ax0 = plt.subplot(121)
    w, h, im = results[0].image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax0.imshow(np_im)
    ax1 = plt.subplot(122, projection='polar')
    for result in results:
        plot_result(result, ax1)

    plt.show()

def viz_circles_datasets(full_sets):
    plt.figure(figsize=(20, 5))

    # Plot the BusyBox
    ax0 = plt.subplot(141)
    w, h, im = full_sets[0][0].image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax0.imshow(np_im)

    plot_nums = [142, 143, 144]
    titles = ['SAGG-RIAC', 'GP-UCB', 'Random']

    for i in range(100):
        for n, (plot_num, title) in enumerate(zip(plot_nums, titles)):
            ax = plt.subplot(plot_num, projection='polar')
            result = full_sets[n][i]
            plot_result(result, ax)
            ax.set_title(title)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            #ax.set_theta_zero_location('N')


    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multiple'], required=True)
    # if mode == single, use to just plot points associated with a single bb
    parser.add_argument('--dataset', type=str)
     # if mode == multiple, use to plot points for different methods (in order: sagg, gpucb, random)
    parser.add_argument('--datasets', nargs='+', type=str)
     # if mode == single
    parser.add_argument('--step', type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.dataset:
        with open(args.dataset, 'rb') as handle:
            data = pickle.load(handle)
        for n in range(0, len(data)-args.step, args.step):
            viz_circles(data[n:n+args.step])
    else:
        all_data = []
        for dataset in args.datasets:
            with open(dataset, 'rb') as handle:
                data = pickle.load(handle)
            all_data += [data]
        #print(len(data))
        first_sets = []
        for dataset in all_data:
            first_sets += [dataset[0:100]]
        viz_circles_datasets(first_sets)
