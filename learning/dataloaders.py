# Given a file setup the dataloaders.
import torch
from torch.utils.data.dataset import Dataset


class PolicyDataset(Dataset):
    def __init__(self):
        super(PolicyDataset, self).__init__()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

def parse_pickle_file(fname):
    """
    Read in the pickle file created by the generate_policy_data file.
    Extract relevant information into vector format.
    :param fname: Name of the pickle file to load.
    :return:
    """
    pass


def setup_data_loaders(batch_size=128, use_cuda=True):
    train_set = PolicyDataset()
    val_set = PolicyDataset()
    test_set = PolicyDataset()

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
    train_loader, val_loader, test_loader = setup_data_loaders(batch_size=16)
