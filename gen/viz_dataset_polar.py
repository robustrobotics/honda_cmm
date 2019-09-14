import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from gen.generator_busybox import BusyBox

def viz_circles(results):
    plt.figure(figsize=(20, 5))
    cmap = plt.get_cmap('viridis')

    # Plot the BusyBox
    ax0 = plt.subplot(121)
    w, h, im = results[0].image_data
    np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)
    ax0.imshow(np_im)
    for result in results:
        ax1 = plt.subplot(122, projection='polar')


        c = cmap(result.net_motion/(result.mechanism_params.params.range/2.0))
        x = [result.policy_params.params.pitch, result.config_goal]
        if x[1] < 0:
            ax1.scatter(x[0] - np.pi, -1 * x[1], c=c, s=10)
        else:
            ax1.scatter(x[0], x[1], c=c, s=10)


    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()

    with open(args.dataset, 'rb') as handle:
        data = pickle.load(handle)
    print(len(data))
    for n in range(0, len(data)-args.step, args.step):
        viz_circles(data[n:n+args.step])