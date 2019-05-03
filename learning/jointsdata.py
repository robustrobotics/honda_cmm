import numpy as np
import pickle

from torch.utils import data


class JointsDataset(data.Dataset):
    def __init__(self, fname):
        self.fname = fname

        self.data = self.create_sets()
        self.datasets = self.data['datasets']

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return len(self.datasets)

    def create_sets(self):
        with open(self.fname, 'rb') as handle:
            data, labs = pickle.load(handle)

        self.n_datasets = len(data)
        self.sample_size = len(data[0])
        self.n_features = len(data[0][0])

        sets = np.zeros((self.n_datasets, self.sample_size, self.n_features),
                        dtype=np.float32)
        labels = []
        for ix, (data, label) in enumerate(zip(data, labs)):
            sets[ix, :, :] = data
            labels.append(label)

        print(labels)
        return {
            "datasets": sets,
            "joints": ["Door_0", "Door_1", "Slider"],
            "labels": np.array(labels)
        }


if __name__ == '__main__':
    dataset = JointsDataset('ns_data/dataset.pkl')
    print(dataset[10])
