import matplotlib.pyplot as plt
import numpy as np


def plot_y_yhat(y, yhat, types, title=''):
    lookup = {'Revolute': 'r',
              'Prismatic': 'b'}
    plt.xlim([0, 0.25])
    plt.ylim([0, 0.25])

    plt.xlabel(r'$y$')
    plt.ylabel(r'$\hat{y}$')

    plt.title(title)

    x = np.linspace(0, 0.5, 100)
    plt.plot(x, x, 'k')

    for n in lookup:
        y1 = [y[ix] for ix, t in enumerate(types) if t == n]
        y2 = [yhat[ix] for ix, t in enumerate(types) if t == n]
        colors = [lookup[n]] * len(y1)
        plt.scatter(y1, y2, s=1, c=colors, label=n)

    plt.legend()
    plt.show()
