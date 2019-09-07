# Given a file setup the dataloaders.
import copy
import numpy as np
import pickle
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from util import util

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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(img.shape) == 3:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        plt.imshow(npimg)
    plt.show()


class PolicyDataset(Dataset):
    def __init__(self, items):
        super(PolicyDataset, self).__init__()
        items = items
        self.items = items

        self.tensors = [torch.tensor(item['params']) for item in items]
        self.configs = [torch.tensor([item['config']]) for item in items]
        self.ys = [torch.tensor([item['y']]) for item in items]

        downsample = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(25),
                                         transforms.Grayscale(),
                                         transforms.ToTensor()])
        tt = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.images = []
        self.downsampled_images = []

        for item in items:
            w, h, im = item['image']
            np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)[:, :, 0:3]
            self.images.append(tt(np_im))
            self.downsampled_images.append(downsample(np_im))
        # imshow(torchvision.utils.make_grid(self.images[0:10]))
        # imshow(torchvision.utils.make_grid(self.downsampled_images[0:10]))

        # this is just for plotting, not learning
        self.delta_vals = [item['delta_vals'] for item in items]

    def __getitem__(self, index):
        return self.items[index]['type'], self.tensors[index], self.configs[index], self.images[index], self.ys[index], self.downsampled_images[index]

    def __len__(self):
        return len(self.items)


def parse_pickle_file(fname=None, data=None):
    """
    Can pass in either a file name or a data dictionary to be parsed.
    Extract relevant information into vector format.
    :param fname: str, name of the pickle file to load.
    :param data: list of util.util.Results to load.
    :return:
    """
    if data is None:
        data = util.read_from_file(fname)

    parsed_data = []
    for entry in data:
        if len(entry) == 0:
            continue
        policy_type = entry.policy_params.type
        if policy_type == 'Prismatic':
            pitch = [entry.policy_params.params.pitch]
            yaw = [entry.policy_params.params.yaw]
            policy_params = pitch + yaw
            mech_params = [entry.mechanism_params.params.range]
        elif policy_type == 'Revolute':
            center = list(entry.policy_params.params.rot_center)
            roll = [entry.policy_params.params.rot_roll]
            pitch = [entry.policy_params.params.rot_pitch]
            orn = list(entry.policy_params.params.rot_orientation)
            radius = list(entry.policy_params.params.rot_radius)
            policy_params = center + roll + pitch + orn + radius
            mech_params = []
        motion = entry.net_motion

        parsed_data.append({
            'type': policy_type,
            'params': policy_params,
            'config': entry.config_goal,
            'image': entry.image_data,
            'y': motion,
            'mech': mech_params,
            'delta_vals': entry.policy_params.delta_values
        })

    return parsed_data


def create_data_splits(data, val_pct=0.15, test_pct=0.15):
    # np.random.shuffle(data)
    n = len(data)
    val_start = int(n*(1-val_pct-test_pct))
    test_start = int(n*(1-test_pct))

    train_data = data[:val_start]
    val_data = data[val_start:test_start]
    test_data = data[test_start:]

    return train_data, val_data, test_data


def setup_data_loaders(fname, batch_size=128, use_cuda=True, small_train=0):
    data = parse_pickle_file(fname)

    # Create datasplits.
    train_data, val_data, test_data = create_data_splits(data)
    if small_train > 0:
        train_data = train_data[:small_train]

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
    train_loader, val_loader, test_loader = setup_data_loaders(fname='prism_rand05_10k.pickle',
                                                               batch_size=16)
    print('Train: {}\tVal: {}\tTest: {}'.format(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    print(train_loader.dataset[0])

    for batch in train_loader:
        print(batch)
