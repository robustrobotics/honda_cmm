import numpy as np
import argparse
import pybullet as p
import pickle
import sys
from test_policy import test_policy
from data.generator import Door, Slider

# range of gains to try
k_lin_range = [-1,5] # sampled in logspace
k_rot_range = [-5,1] # sampled in logspace
d_lin_range = [-5,4] # sampled in logspace
d_rot_range = [-5,1] # sampled in logspace
add_dist_range = [0.,.1]
p_err_eps_range = [.001, .1]
delta_pos_range = [.001, .1]

def learn_gains(file_name, n_samples, viz, debug):
    results = []
    for mech_type in [Door, Slider]:
        type = 'door' if mech_type == Door else 'slider'
        for i in range(n_samples):
            sys.stdout.write("\rProcessing %s %i/%i" % (type, i+1, n_samples))

            k_lin = np.power(10., np.random.uniform(*k_lin_range))
            k_rot = np.power(10., np.random.uniform(*k_rot_range))
            d_lin = np.power(10., np.random.uniform(*d_lin_range))
            d_rot = np.power(10., np.random.uniform(*d_rot_range))
            k = [k_lin, k_rot]
            d = [d_lin, d_rot]
            add_dist = np.random.uniform(*add_dist_range)
            p_err_eps = np.random.uniform(*p_err_eps_range)
            delta_pos = np.random.uniform(*delta_pos_range)

            results += test_policy(viz, debug, 1, True, k, d, add_dist, p_err_eps, delta_pos)

            # save to pickle (keep overwriting latest file in case it crashes)
            fname = file_name + '.pickle'
            with open(fname, 'wb') as handle:
                pickle.dump(results, handle)

def plot_from_file(file_name):
    fname = file_name + '.pickle'
    with open(fname, 'rb') as handle:
        results = pickle.load(handle)

    # plot results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    cm = plt.cm.get_cmap('copper')

    min_time = float('inf')
    max_time = 0
    for result in results:
        if result.waypoints_reached>.5:
            # for some reason the time is negative sometimes...
            if result.duration > 0:
                if result.duration>max_time:
                    max_time = result.duration
                if result.duration<min_time:
                    min_time = result.duration

    for result in results:
        wr = result.waypoints_reached
        time = result.duration
        a = ax0.scatter([result.k[0]], [result.d[0]], cmap=cm, c=[wr], s=2, vmin=0, vmax=1) # s is markersize
        b = ax1.scatter([result.k[1]], [result.d[1]], cmap=cm, c=[wr], s=2, vmin=0, vmax=1)
    ks = np.power(10.,np.linspace(k_lin_range[0], k_lin_range[1],1000))
    mass = 1.5
    ds_critically_damped = np.sqrt(4*mass*ks)
    ax0.plot(ks, ds_critically_damped, label='critically damped')

    ax0.set_xlabel('Linear K')
    ax0.set_ylabel('Linear D')
    ax0.set_title('Time before Reached Goal or Timeout for Doors\n(for > .5 of Waypoints Reached)')
    ax0.legend()

    ax1.set_xlabel('Angular K')
    ax1.set_ylabel('Angular D')
    ax1.set_title('Time before Reached Goal or Timeout for Doors\n(for > .5 of Waypoints Reached)')

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax0.set_xlim(*np.power(10.,k_lin_range))
    ax0.set_ylim(*np.power(10.,d_lin_range))
    ax1.set_xlim(*np.power(10.,k_rot_range))
    ax1.set_ylim(*np.power(10.,d_rot_range))

    fig0.colorbar(a)
    fig1.colorbar(b)

    plt.show()
    input('hit enter when done\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--plot', type=str) # give filename (without .pickle)
    parser.add_argument('--file', type=str) # give filename (without .pickle)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if not args.plot:
        learn_gains(args.file, args.n_samples, args.viz, args.debug)
        plot_from_file(args.file)
    else:
        plot_from_file(args.plot)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting...')
