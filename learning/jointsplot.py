import numpy as np

from matplotlib import pyplot as plt


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


def contexts_by_moment(contexts, moments, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    contexts = np.array(contexts).reshape(-1, 3)
    cax = ax.scatter(contexts[:, 0], contexts[:, 1], contexts[:, 2],
                     c=moments[:len(contexts)])
    fig.colorbar(cax)

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
                    right='off', left='off', labelleft='off')
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
