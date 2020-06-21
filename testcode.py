# from argparse import Namespace
# from actions.policies import Policy
# from learning.models.nn_disp_pol_vis import DistanceRegressor as NNPolVis
# from learning.gp.explore_single_bb import create_gpucb_dataset
# from utils import util
# from learning.dataloaders import setup_data_loaders, parse_pickle_file

# policy_types = ['Prismatic', 'Revolute']
# net = NNPolVis(policy_names=policy_types,
#                policy_dims=Policy.get_param_dims(policy_types),
#                hdim=16,
#                im_h=53,  # 154, Note these aren't important for the SpatialAutoencoder
#                im_w=115,  # 205,
#                image_encoder='spatial')
# train_args = Namespace(n_gp_samples=500, bb_fname='', mech_types=['slider'], plot=False, urdf_num=0,
#                        fname='', nn_fname='', plot_dir='', debug=False, random_policies=False, stochastic=False)
# new_data = create_gpucb_dataset(100, 1, train_args, net)
# print('NEW CREATED DATA')
# print(len(new_data[0][0]))
# print(len(new_data[0][1]))
# print(parse_pickle_file(new_data[0]))

filename = 'honda_cmm/continual/test_train_models/cont_models/torch_models/model_100M26500.pt'
start = filename.find('_100M')
print(int(filename[start+5:-3]))
