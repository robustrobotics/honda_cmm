import pickle
import argparse
import os
import numpy as np
from learning.gp.explore_single_bb import viz_circles
from learning.gp.evaluate_models import get_models
from utils import util
from utils.setup_pybullet import setup_env
from gen.generator_busybox import BusyBox
from actions.policies import Policy


def viz_models(args):
    with open(args.bb_fname, 'rb') as handle:
        bb_data = pickle.load(handle)

    # TODO: model file names use n_interactions*num_bbs and n_interactions is
    # hard coded to 100 for now, so multipy by 100
    models = get_models(100*args.L, args.models_path)
    for model in models:
        for ix, bb_result in enumerate(bb_data[:args.n_bbs]):
            if args.debug:
                print('BusyBox', ix)

            # load model
            nn = util.load_model(model, args.hdim, use_cuda=False)

            # get image data
            bb = BusyBox.bb_from_result(bb_result)
            mech = bb._mechanisms[0]
            image_data, gripper = setup_env(bb, False, False, args.no_gripper)

            # generate plots
            viz_circles(image_data, mech, nn=nn, bb_i=ix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--L',
        type=int,
        required=True,
        help='what number of training Mechanisms to evaluate')
    parser.add_argument(
        '--n-bbs',
        type=int,
        help='number of BusyBoxes to visualize')
    parser.add_argument(
        '--bb-fname',
        default='',
        help='path to file of BusyBoxes to interact with')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='use to enter debug mode')
    parser.add_argument(
        '--hdim',
        type=int,
        default=16,
        help='hdim of supplied model(s), used to load model file')
    parser.add_argument(
        '--models-path',
        help='path to model files')
    parser.add_argument(
        '--no-gripper',
        help='use to apply foce directly to handles')
    args = parser.parse_args()

    if args.debug:
        import pdb; pdb.set_trace()

    viz_models(args)
