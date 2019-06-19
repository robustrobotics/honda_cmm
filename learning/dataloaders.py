# Given a file setup the dataloaders.
import pickle
import torch
from torch.utils.data.dataset import Dataset


class PolicyDataset(Dataset):
    def __init__(self, items):
        super(PolicyDataset, self).__init__()
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        return item['type'], item['params'], item['y']

    def __len__(self):
        return len(self.items)


def parse_pickle_file(fname):
    """
    Read in the pickle file created by the generate_policy_data file.
    Extract relevant information into vector format.
    :param fname: Name of the pickle file to load.
    :return:
    """
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)

    parsed_data = []
    for entry in data:
        if len(entry) == 0:
            continue
        policy_type = entry[0].policy_params.type
        if policy_type == 'Prismatic':
            pos = list(entry[0].policy_params.params.rigid_position)
            orn = list(entry[0].policy_params.params.rigid_orientation)
            dir = list(entry[0].policy_params.params.prismatic_dir)
            policy_params = pos + orn + dir
        elif policy_type == 'Revolute':
            center = list(entry[0].policy_params.params.rot_center)
            axis = list(entry[0].policy_params.params.rot_axis)
            orn = list(entry[0].policy_params.params.rot_orientation)
            radius = list(entry[0].policy_params.params.rot_radius)
            policy_params = center + axis + orn + radius
        motion = entry[0].motion

        parsed_data.append({
            'type': policy_type,
            'params': policy_params,
            'y': motion
        })

    return parsed_data


def create_data_splits(data, val_pct=0.15, test_pct=0.15):
    n = len(data)
    val_start = int(n*(1-val_pct-test_pct))
    test_start = int(n*(1-test_pct))

    train_data = data[:val_start]
    val_data = data[val_start:test_start]
    test_data = data[test_start:]

    return train_data, val_data, test_data



def setup_data_loaders(fname, batch_size=128, use_cuda=True):
    data = parse_pickle_file(fname)

    # Create datasplits.
    train_data, val_data, test_data = create_data_splits(data)

    # TODO: Populate dataset objects

    train_set = PolicyDataset(train_data)
    val_set = PolicyDataset(val_data)
    test_set = PolicyDataset(test_data)

    kwargs = {'num_workers': 0,
              'pin_memory': use_cuda}

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              **kwargs)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = setup_data_loaders(fname='smalltest.pickle',
                                                               batch_size=16)
    print('Train: {}\tVal: {}\tTest: {}'.format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    print(train_loader.dataset[0])
