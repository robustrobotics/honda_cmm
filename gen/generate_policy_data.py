import sys
import argparse
from test_policy import test_policy
from util import util

results = []
def generate_data(n_samples, viz, debug, git_hash, urdf_num):
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        results.extend(test_policy(viz=viz, debug=debug, max_mech=1, random=True, \
                        git_hash=git_hash, tag='_policy_'+str(urdf_num)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--fname', type=str) # give filename (without .pickle)
    parser.add_argument('--test-read', action='store_true')
    parser.add_argument('--save-git', action='store_true')
    # if running multiple tests, give then a urdf_num so correct urdf read from/written to
    parser.add_argument('--urdf-num', type=int, default=0)
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()


    if args.test_read:
        data = util.read_from_file(args.fname)
    else:
        try:
            git_hash = None
            if args.save_git:
                import git
                repo = git.Repo(search_parent_directories=True)
                git_hash = repo.head.object.hexsha
            generate_data(args.n_samples, args.viz, args.debug, git_hash, args.urdf_num)
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
