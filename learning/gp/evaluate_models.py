import pickle
import argparse
import numpy as np
from learning.gp.explore_single_bb import create_single_bb_gpucb_dataset, test_model
from utils import util
def evaluate_models(n_interactions, n_bbs, args, use_cuda=False):
    with open(args.bb_fname, 'rb') as handle:
        data = pickle.load(handle)

    results = []
    for model in args.models:
        avg_regrets, final_regrets = [], []
        for ix, bb_result in enumerate(data[:n_bbs]):
            print('BusyBox', ix)
            dataset, avg_regret, gp = create_single_bb_gpucb_dataset(bb_result, n_interactions, model, args)
            nn = util.load_model(model, args.hdim, use_cuda=False)
            regret = test_model(gp, bb_result, args, nn, use_cuda=use_cuda, urdf_num=args.urdf_num)
            avg_regrets.append(avg_regret)
            print('Average Regret:', avg_regret)
            final_regrets.append(regret)
            print('Test Regret   :', regret)
        print('Results')
        print('Average Regret:', np.mean(avg_regrets))
        print('Final Regret  :', np.mean(final_regrets))
        res = {'model': model,
               'avg': np.mean(avg_regrets),
               'final': np.mean(final_regrets),
               'regrets': final_regrets}
        results.append(res)
        results_fname = 'regret_results_%s_t%d_n%d.pickle'
        print(results_fname % (args.eval, n_interactions, n_bbs))
        with open(results_fname % (args.eval, n_interactions, n_bbs), 'wb') as handle:
            pickle.dump(results, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n-gp-samples',
        type=int,
        default=500,
        help='number of samples to use when fitting a GP to data')
    parser.add_argument(
        '--T',
        type=int,
        help='number of interactions within a single BusyBox during evaluation time')
    parser.add_argument(
        '--N',
        type=int,
        help='number of BusyBoxes to interact with during evaluation time')
    parser.add_argument(
        '--eval',
        type=str,
        choices=['active_nn', 'gpucb_nn', 'random_nn', 'test_good', 'test_bad'],
        default='',
        help='evaluation type')
    parser.add_argument(
        '--urdf-num',
        default=0,
        help='number to append to generated urdf files. Use if generating multiple datasets simultaneously.')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--plot',
        action='store_true',
        help='use to generate polar plots durin GP-UCB interactions')
    parser.add_argument(
        '--nn-fname',
        default='',
        help='path to NN to initialize GP-UCB interactions')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--hdim',
        type=int,
        default=16,
        help='hdim of supplied model(s), used to load model file'
    )
    parser.add_argument(
        '--models',
        nargs='*',
        help='list of NN model files to evaluate with GP-UCB method')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    evaluate_models(args.T, args.N, args, use_cuda=False)
