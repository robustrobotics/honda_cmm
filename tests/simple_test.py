"""
A quick sanity check that the main exported functions of the honda_cmm repo are
working. This test file is missing tests for the gen.active_prior and
learning.train_active modules.
"""
import numpy as np
import os
from utils.util import read_from_file, compare_results, load_model, compare_models
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import torch
from actions.systematic import execute_systematic
from gen.generate_policy_data import generate_dataset
from learning.gp.explore_single_bb import create_gpucb_dataset, evaluate_models
from learning.train import train_eval

BB_FILE = 'tests/bbs_file.pickle'
SEED = 1

def test_systematic():
    """ Tests the systematic module by comparing the output pickle file to a
    version that was output when the code was functioning correctly.
    """
    working_output_file = 'tests/systematic_n2_t5.pickle'
    test_output_file = 'tests/test_systematic.pickle'
    args = Namespace(N=2,
            T=5,
            bb_file=BB_FILE,
            fname=test_output_file,
            viz=None,
            debug=None)
    execute_systematic(args) # saves a dictionary to file which can be compared with ==
    working_output = read_from_file(working_output_file)
    test_output = read_from_file(test_output_file)
    if working_output == test_output:
        os.remove(test_output_file)
    else:
        assert False, 'the actions.systematic module is broken'


def test_gen_dataset():
    """ Tests the code for generating random datasets by comparing the output
    pickle file to a version that was output when the code was functioning
    correctly.
    """
    np.random.seed(SEED)
    working_output_file = 'tests/random_dataset.pickle'
    args = Namespace(viz=None,
            debug=None,
            n_samples=5,
            n_bbs=2,
            max_mech=1,
            fname=None,
            urdf_num=0,
            match_policies=True,
            randomness=1.0,
            goal_config=None,
            bb_file=BB_FILE)
    test_output = generate_dataset(args, None)
    working_output = read_from_file(working_output_file)
    assert compare_results(test_output, working_output), \
        'the gen.generate_policy_data module is broken'

def test_gp_gen_dataset():
    """ Tests the code for generating a GP-UCB dataset by comparing the output
    pickle file to a version that was output when the code was functioning
    correctly.
    """
    np.random.seed(SEED)
    working_output_file = 'tests/gp_ucb_dataset.pickle'
    test_output_file = 'tests/test_gpucb_dataset.pickle'
    args = Namespace(n_train=10,
            T=None,
            M=5,
            L=2,
            N=None,
            eval='',
            urdf_num=0,
            bb_fname=BB_FILE,
            plot=None,
            nn_fname='',
            fname=test_output_file,
            debug=None,
            models=None)
    create_gpucb_dataset(args.M, args.L, args)
    test_output = read_from_file(test_output_file)
    working_output = read_from_file(working_output_file)
    if compare_results(test_output, working_output):
        os.remove(test_output_file)
    else:
        assert False, 'the learning.gp.explore_single_bb module is broken'
        

def test_train():
    """ Tests the training code to see if the same model is output from the same
    input (given a random seed).
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    working_model_file = 'tests/train/torch_models/model_nrun_0.pt'
    working_data_file = 'tests/random_dataset.pickle'
    test_model_dir = 'tests/test_train/'
    args = Namespace(batch_size=3,
            hdim=3,
            n_epochs=3,
            val_freq=3,
            mode='normal',
            use_cuda=False,
            debug=None,
            data_fname=working_data_file,
            save_dir=test_model_dir,
            n_train_min=0,
            n_train_max=0,
            step=0,
            image_encoder='spatial',
            n_runs=1,
            pviz=None)
    writer = SummaryWriter('tests/runs')
    test_model_file = test_model_dir+'model.pt'
    train_eval(args, args.hdim, args.batch_size, args.pviz, test_model_file, writer)
    working_model = load_model(working_model_file, hdim=args.hdim)
    test_model = load_model(test_model_file, hdim=args.hdim)
    if compare_models(working_model, test_model):
        os.remove(test_model_file)
    else:
        assert False, 'the learning.train module is broken'

if __name__ == '__main__':
    test_systematic()
    test_gen_dataset()
    test_gp_gen_dataset()
    test_train()
