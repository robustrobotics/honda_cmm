# Given a file setup the dataloaders.
import copy
import numpy as np
import pickle
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, items, batch_size):
        self.batch_size = batch_size

        self.types = []
        self.sorted_indices = {}
        for ix, item in enumerate(items):
            k = item['type']
            if k not in self.types:
                self.types.append(k)
                self.sorted_indices[k] = []
            self.sorted_indices[k].append(ix)

    def __iter__(self):
        # Shuffle each policy class individually and create an iterator for each.
        iters = {}
        valid_types = copy.deepcopy(self.types)
        for k in self.sorted_indices:
            np.random.shuffle(self.sorted_indices[k])
            iters[k] = iter(self.sorted_indices[k])

        # Randomly choose a class until we've iterated through everything.
        while len(valid_types) > 0:
            # Choose a policy type.
            k = np.random.choice(valid_types)

            # Create a batch of batch_size.
            batch = []
            for _ in range(self.batch_size):
                # Check if we've used all the data from a policy iterator.
                try:
                    ix = next(iters[k])
                except:
                    valid_types.remove(k)
                    break
                batch.append(ix)
            # Yield the batch.
            if len(batch) > 0:
                yield batch

    def __len__(self):
        """
        Batches consist of a single policy type.
        :return: The number of batches in the dataset.
        """
        n = 0
        for v in self.sorted_indices.values():
            n += (len(v) + self.batch_size - 1) // self.batch_size
        return n


class PolicyDataset(Dataset):
    def __init__(self, items):
        super(PolicyDataset, self).__init__()
        self.items = items
        self.tensors = [torch.tensor(item['params']) for item in items]
        self.configs = [torch.tensor([item['config']]) for item in items]
        self.ys = [torch.tensor([item['y']]) for item in items]

    def __getitem__(self, index):
        return self.items[index]['type'], self.tensors[index], self.configs[index], self.ys[index]

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
        policy_type = entry.policy_params.type
        if policy_type == 'Prismatic':
            pos = list(entry.policy_params.params.rigid_position)
            orn = list(entry.policy_params.params.rigid_orientation)
            dir = list(entry.policy_params.params.prismatic_dir)
            range = [entry.mechanism_params.params.range]
            policy_params = pos + orn + dir + range
        elif policy_type == 'Revolute':
            center = list(entry.policy_params.params.rot_center)
            axis = list(entry.policy_params.params.rot_axis)
            orn = list(entry.policy_params.params.rot_orientation)
            radius = list(entry.policy_params.params.rot_radius)
            flipped = [entry.mechanism_params.params.flipped]
            door_size = list(entry.mechanism_params.params.door_size)
            policy_params = center + axis + orn + radius + door_size + flipped
        motion = entry.motion

        parsed_data.append({
            'type': policy_type,
            'params': policy_params,
            'config': entry.config_goal,
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
                                               batch_sampler=CustomSampler(train_set.items, batch_size),
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_sampler=CustomSampler(val_set.items, batch_size),
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_sampler=CustomSampler(test_set.items, batch_size),
                                              **kwargs)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = setup_data_loaders(fname='smalltest.pickle',
                                                               batch_size=16)
    print('Train: {}\tVal: {}\tTest: {}'.format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    print(train_loader.dataset[0])

    for batch in train_loader:
        print(batch)