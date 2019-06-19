import pickle
import sys
import argparse
from test_policy import test_policy

samples = []
def generate_data(n_samples, viz, debug):
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        samples.append(test_policy(viz=viz, debug=debug, max_mech=1, random=True))

def write_to_file(file_name):
    # save to pickle
    fname = file_name + '.pickle'
    with open(fname, 'wb') as handle:
        pickle.dump(samples, handle)
    print('\nwrote dataset to '+fname)

def test_read(file_name):
    fname = file_name + '.pickle'
    with open(fname, 'rb') as handle:
        results = pickle.load(handle)
        print('successfully read in '+fname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--fname', type=str) # give filename (without .pickle)
    parser.add_argument('--test-read', action='store_true')
    args = parser.parse_args()

    try:
        if args.debug:
            import pdb; pdb.set_trace()

        if not args.test_read:
            generate_data(args.n_samples, args.viz, args.debug)
            write_to_file(args.fname)
        else:
            test_read(args.fname)
    except KeyboardInterrupt:
        # if Ctrl+C write to pickle
        write_to_file(args.fname)
        print('Exiting...')
    except:
        # if crashes write to pickle
        write_to_file(args.fname)

        # for post-mortem debugging since can't run module from command line in pdb.pm() mode
        import traceback, pdb, sys
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
