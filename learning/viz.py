import matplotlib.pyplot as plt
import numpy as np
import torch
from learning.dataloaders import setup_data_loaders
from learning.nn_disp_pol_vis import DistanceRegressor as NNPolVis

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



def plot_q_yhat(data_loader, net, n_samples=50):

    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    for bx in range(len(data_loader.dataset)):
        k, x, qo, im, y = data_loader.dataset[bx]
        x = x.unsqueeze(0).cuda()
        im = im.unsqueeze(0).cuda()

        qs = np.linspace(-0.5, 0.5, n_samples)
        ys = []
        for q in qs:
            q = torch.tensor([[q]]).cuda()
            yhat = net.forward(k, x, q, im)
            ys.append(yhat.item())

        r = data_loader.dataset.items[bx]['mech'][0]/2
        print(r)

        row = bx // 4
        col = bx % 4

        axes[row][col].plot([-0.5, -r], [r, r], c='r', label='true')
        axes[row][col].plot([-r, 0], [r, 0], c='r')
        axes[row][col].plot([0, r], [0, r], c='r')
        axes[row][col].plot([r, 0.5], [r, r], c='r')
        axes[row][col].scatter([qo.item()], [y.item()], c='g')

        axes[row][col].set_xlim(-0.25, 0.25)
        axes[row][col].set_ylim(0, 0.25)
        axes[row][col].set_xlabel('q')
        axes[row][col].set_ylabel('y')
        #plt.plot([-r, -r], [0, 0.25], c='r')
        #plt.plot([r, r], [0, 0.25], c='r')
        axes[row][col].set_title('r=%.2f' % r, fontsize=8)
        axes[row][col].plot(qs, ys, 'b', label='pred')

        if bx >= 15: break
    axes[0][3].legend()
    plt.show()


if __name__ == '__main__':
    train_set, val_set, test_set = setup_data_loaders(fname='data/newdata.pickle',
                                                      batch_size=1,
                                                      small_train=0)

    net = NNPolVis(policy_names=['Prismatic', 'Revolute'],
                   policy_dims=[9, 12],
                   hdim=16,
                   im_h=53,
                   im_w=115,
                   kernel_size=3).cuda()
    device = torch.device('cuda')
    net.load_state_dict(torch.load('data/models/best_prism_only_color._0.pt', map_location=device))
    net.eval()
    plot_q_yhat(test_set, net)