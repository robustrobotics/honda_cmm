"""
Generate a dataset of door images and their respective radii.
"""
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

from gen.generator_busybox import BusyBox, Door
from utils import util
from utils.setup_pybullet import setup_env


class ImageDataset(Dataset):
    def __init__(self, items):
        """ Load a Dataset with regression from a single image.
        :param items: A list of (img, radius) pairs.
        """
        super(ImageDataset, self).__init__()
        self.items = items
        self.images = []
        self.ys = [torch.tensor([item[1]]) for item in items]

        tt = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        for image_data, _ in items:
            w, h, im = image_data
            np_im = np.array(im, dtype=np.uint8).reshape(h, w, 3)[:, :, 0:3]
            self.images.append(tt(np_im))

    def __getitem__(self, index):
        return self.images[index], self.ys[index]

    def __len__(self):
        return len(self.items)


def create_data_splits(data, val_pct=0.1, test_pct=0.1):
    n = len(data)
    val_start = int(n*(1-val_pct-test_pct))
    test_start = int(n*(1-test_pct))

    train_data = data[:val_start]
    val_data = data[val_start:test_start]
    test_data = data[test_start:]

    return train_data, val_data, test_data


def setup_data_loaders(data, batch_size=128, use_cuda=True):
    """
    :param data:
    :param batch_size: If -1, set the batch size to the size of the dataset.
    """
    kwargs = {'num_workers': 0,
              'pin_memory': use_cuda}

    # Create datasplits.
    train_data, val_data, test_data = create_data_splits(data, val_pct=.2)
    random.Random(0).shuffle(train_data)
    random.Random(0).shuffle(val_data)
    random.Random(0).shuffle(test_data)

    # TODO: Populate dataset objects

    train_set = ImageDataset(train_data)
    val_set = ImageDataset(val_data)
    test_set = ImageDataset(test_data)

    train_batch = val_batch = test_batch = batch_size
    if batch_size == -1:
        train_batch = len(train_data)
        val_batch = len(val_data)
        test_batch = len(test_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch,
                                               **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                             batch_size=val_batch,
                                             **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=test_batch,
                                              **kwargs)
    return train_loader, val_loader, test_loader


def generate_dataset(N):
    """ Generate a dataset of (I, r) pairs where I is an image of
        a door and r is its radius.
    :param N: Number of doors to generate.
    """
    dataset = []
    for _ in range(N):
        bb = BusyBox.generate_random_busybox(max_mech=1,
                                             mech_types=[Door],
                                             urdf_tag=N)
        mechanism_params = bb._mechanisms[0].get_mechanism_tuple()
        radius = mechanism_params.params.door_size[0]

        image_data, gripper = setup_env(bb=bb, 
                                        viz=False, 
                                        debug=False, 
                                        use_gripper=False, 
                                        show_im=False)
        
        dataset.append((image_data, radius))
    
    return dataset        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--fname', type=str, required=True)
    args = parser.parse_args()

    dataset = generate_dataset(args.N)


    with open(args.fname, 'wb') as handle:
        pickle.dump(dataset, handle)

    with open(args.fname, 'rb') as handle:
        dataset = pickle.load(handle)

    train, val, test = setup_data_loaders(dataset, batch_size=-1, use_cuda=False)

    for x, y in train:
        print(x.shape, y.shape)



    