from utils import util
data = util.read_from_file('output.pickle')
for bb_data in data:
    data_point = bb_data[0]
    image_data = data_point.image_data
    policy_data = data_point.policy_params
    motion = data_point.net_motion
    print(policy_data.type)
    print(policy_data.params['pitch'])
    print(motion)