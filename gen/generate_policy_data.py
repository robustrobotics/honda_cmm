import pickle
import sys
import argparse
from test_policy import test_policy

def generate_data(file_name, n_samples, viz, debug):
    samples = []
    for i in range(n_samples):
        sys.stdout.write("\rProcessing sample %i/%i" % (i+1, n_samples))
        samples += test_policy(viz=viz, debug=debug, max_mech=1, random=True)

        # save to pickle (keep overwriting latest file in case it crashes)
        fname = file_name + '.pickle'
        with open(fname, 'wb') as handle:
            pickle.dump(samples, handle)
    print('done generating dataset')

def test_read(file_name):
    fname = file_name + '.pickle'
    with open(fname, 'rb') as handle:
        results = pickle.load(handle)
        print('successfully read in pickle file')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n-samples', type=int, default=5)
    parser.add_argument('--fname', type=str) # give filename (without .pickle)
    parser.add_argument('--test-read', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    if not args.test_read:
        generate_data(args.fname, args.n_samples, args.viz, args.debug)
    else:
        test_read(args.fname)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print('Exiting...')
