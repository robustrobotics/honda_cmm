import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_contexts(contexts, labels, joints, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    contexts = np.array(contexts).reshape(-1, 2)
    n = len(contexts)

    labels = labels[:n]
    ix = [np.where(labels == label)
          for i, label in enumerate(joints)]
    colors = [
        'indianred',
        'forestgreen',
        'blue'
    ]

    for label, i in enumerate(ix):
        ax.scatter(contexts[i][:, 0], contexts[i][:, 1],
                   label=joints[label].title(),
                   color=colors[label],
                   s=1)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                   right='off', left='off', labelleft='off')
    plt.legend(loc='upper left')
    plt.tight_layout()


    if savepath is not None:
        plt.savefig(savepath)
    plt.close()


def sample_configurations(true_xs, sampled_xs, recon_xs, savefolder=None, suffix=''):
    true_xs = np.array(true_xs)
    sampled_xs = np.array(sampled_xs)
    recon_xs = np.array(recon_xs)

    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    for ix in range(25):#true_xs.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(sampled_xs[ix, :, 0],
                   sampled_xs[ix, :, 1],
                   sampled_xs[ix, :, 2],
                   color='red',
                   s=1)
        ax.scatter(true_xs[ix, :, 0],
                   true_xs[ix, :, 1],
                   true_xs[ix, :, 2],
                   color='green',
                   s=1)
        ax.scatter(recon_xs[ix, :, 0],
                   recon_xs[ix, :, 1],
                   recon_xs[ix, :, 2],
                   color='blue',
                   s=1)
        #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
        #                right='off', left='off', labelleft='off')
        #plt.tight_layout()
        ax.set_xlim3d(-0.25, 0.25)
        ax.set_ylim3d(-0.25, -0.1)
        ax.set_zlim3d(0, 0.3)
        if savefolder is not None:
            plt.savefig('{0}/{1}{2}.png'.format(savefolder, ix, suffix))
        plt.close()

