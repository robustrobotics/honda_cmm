import numpy as np
import argparse
import pybullet as p
import sys
from test_policy import test_policy
from gen.generator_busybox import Door, Slider
from util import util

# range of gains to try
k_lin_range = [-1,5] # sampled in logspace
k_rot_range = [-5,1] # sampled in logspace
d_lin_range = [-5,4] # sampled in logspace
d_rot_range = [-5,1] # sampled in logspace
add_dist_range = [0.,.1]
p_err_thresh_range = [.001, .1]
p_delta_range = [.001, .1]

results = []

def learn_gains(file_name, n_samples, viz, debug, git_hash, urdf_num):
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        k_lin = np.power(10., np.random.uniform(*k_lin_range))
        k_rot = np.power(10., np.random.uniform(*k_rot_range))
        d_lin = np.power(10., np.random.uniform(*d_lin_range))
        d_rot = np.power(10., np.random.uniform(*d_rot_range))
        k = [k_lin, k_rot]
        d = [d_lin, d_rot]
        add_dist = np.random.uniform(*add_dist_range)
        p_err_thresh = np.random.uniform(*p_err_thresh_range)
        p_delta = np.random.uniform(*p_delta_range)
        results.extend(test_policy(viz=viz, debug=debug, max_mech=1, random=True,\
                        k=k, d=d, add_dist=add_dist, p_err_thresh=p_err_thresh, \
                        p_delta=p_delta, tag='_gains_'+str(urdf_num)))

def plot_from_file(file_name):
    plot_data = util.read_from_file(file_name)

    # plot results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.ion()
    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()
    cm = plt.cm.get_cmap('copper')

    for data_point in plot_data:
        wr = data_point.waypoints_reached
        a = ax0.scatter([data_point.control_params.k[0]], [data_point.control_params.d[0]], \
                            cmap=cm, c=[wr], s=2, vmin=0, vmax=1) # s is markersize
        b = ax1.scatter([data_point.control_params.k[1]], [data_point.control_params.d[1]], \
                            cmap=cm, c=[wr], s=2, vmin=0, vmax=1)

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
    input('\nhit enter when done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--plot', action='store_true') # give filename (without .pickle)
    parser.add_argument('--fname', type=str) # give filename (without .pickle)
    parser.add_argument('--save-git', action='store_true')
    # if running multiple tests, give then a urdf_num so correct urdf read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if args.plot:
        plot_from_file(args.fname)
    else:
        try:
            git_hash = None
            if args.save_git:
                import git
                repo = git.Repo(search_parent_directories=True)
                git_hash = repo.head.object.hexsha
            learn_gains(args.fname, args.n_samples, args.viz, args.debug, git_hash, args.urdf_num)
            util.write_to_file(args.fname, results)
        except KeyboardInterrupt:
            # if Ctrl+C write to pickle
            util.write_to_file(args.fname, results)
            print('Exiting...')
        except:
            # if crashes write to pickle
            util.write_to_file(args.fname, results)

            # for post-mortem debugging since can't run module from command line in pdb.pm() mode
            import traceback, pdb, sys
            traceback.print_exc()
            print('')
            pdb.post_mortem()
            sys.exit(1)
