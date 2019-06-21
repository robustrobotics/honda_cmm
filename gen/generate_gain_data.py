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
        results.extend(test_policy(viz=viz, debug=debug, max_mech=1, random_policy=False,\
                        k=k, d=d, add_dist=add_dist, p_err_thresh=p_err_thresh, \
                        p_delta=p_delta, tag='_gains_'+str(urdf_num)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--fname', type=str, required=True) # give filename (without .pickle)
    # if running multiple tests, give then a urdf_num so correct urdf read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    try:
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            git_hash = repo.head.object.hexsha
        except:
            git_hash = None
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
