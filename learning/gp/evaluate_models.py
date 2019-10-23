import pickle
import argparse
import os
import numpy as np
from learning.gp.explore_single_bb import create_single_bb_gpucb_dataset, test_model
from utils import util

def get_models(L, models_path):
    all_files = os.walk(models_path)
    models = []
    for root, subdir, files in all_files:
        for file in files:
            if file[-3:] == '.pt' and str(L)+'.pt' in file:
                full_path = root+'/'+file
                models.append(full_path)
    return models

def evaluate_models(n_interactions, n_bbs, args, use_cuda=False):
    with open(args.bb_fname, 'rb') as handle:
        bb_data = pickle.load(handle)

    all_results = []
    for L in range(1000, 10001, 1000):
        models = get_models(L, args.models_path)
        all_model_results = []
        L_final_regrets = []
        for model in models:
            model_final_regrets = []
            for ix, bb_result in enumerate(bb_data[:n_bbs]):
                print('BusyBox', ix)
                dataset, avg_regret, gp = create_single_bb_gpucb_dataset(bb_result, n_interactions, model, args)
                nn = util.load_model(model, args.hdim, use_cuda=False)
                regret = test_model(gp, bb_result, args, nn, use_cuda=use_cuda, urdf_num=args.urdf_num)
                model_final_regrets.append(regret)
                L_final_regrets.append(regret)
                if args.debug:
                    print('Average Regret:', avg_regret)
                    print('Test Regret   :', regret)
            if args.debug:
                print('Results')
                #print('Average Regret:', np.mean(avg_regrets))
                print('Final Regret  :', np.mean(model_final_regrets))
            model_result = {'model': model,
                           'final': np.mean(model_final_regrets)}
            all_model_results.append(model_result)
        if len(models) > 0:
            L_result = {'L': L/100, # TODO: this shouldn't be hard coded to 100 interactions per BB
                        'final': np.mean(L_final_regrets),
                        'model_data': all_model_results}
            all_results.append(L_result)
    results_fname = 'regret_results_%s_%dT_%dN.pickle'
    #print(results_fname % (args.eval, n_interactions, n_bbs))
    with open(results_fname % (args.eval, n_interactions, n_bbs), 'wb') as handle:
        pickle.dump(all_results, handle)


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
        '--models-path',
        help='path to model files')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    evaluate_models(args.T, args.N, args, use_cuda=False)
