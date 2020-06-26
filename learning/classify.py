import torch
import torch.nn as nn
from utils.util import load_model, read_from_file
from learning.dataloaders import setup_data_loaders, parse_pickle_file


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        im_dim, im_h_dim = 3, 16
        self.c, self.h, self.w = 3, 59, 58
        kernel_size, padding = 3, 1
        self.conv1 = nn.Conv2d(in_channels=im_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
        self.tanh = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding)

        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(9*im_h_dim, im_h_dim)
        self.lin2 = nn.Linear(im_h_dim, 1)
        self.dropout = nn.Dropout(0.)

    def forward(self, im):
        im = im[:, :, ::2, ::2]
        im = im.view(-1, self.c, self.h, self.w)
        im = self.conv1(im)
        im = self.dropout(im)
        im = self.tanh(im)
        im = self.maxpool(im)
        im = self.conv2(im)
        im = self.dropout(im)
        im = self.tanh(im)
        im = self.maxpool(im)
        im = self.conv3(im)
        im = self.dropout(im)
        im = self.tanh(im)
        im = self.maxpool(im)
        im = self.conv4(im)
        im = self.dropout(im)
        im = self.tanh(im)
        im = self.maxpool(im)
        im = self.flatten(im)
        im = self.lin1(im)
        im = self.dropout(im)
        im = self.tanh(im)
        return self.sigmoid(self.lin2(im))

def get_datasets(results, n_train=80):
    train_results = [[bb[0]] for bb in raw_results[0:n_train]]
    train_results = [item for sublist in train_results for item in sublist]
    train_data = parse_pickle_file(train_results)
    train_set  = setup_data_loaders(data=train_data, batch_size=16, single_set=True)

    val_results = [[bb[0]] for bb in raw_results[n_train:]]
    val_results = [item for sublist in val_results for item in sublist]
    val_data = parse_pickle_file(val_results)
    val_set  = setup_data_loaders(data=val_data, batch_size=16, single_set=True)

    return train_set, val_set


def evaluate(net, dataset):
    correct, total = 0., 0.
    with torch.no_grad():
        for policy_type, theta, im, y, mech in dataset:
            im = im.cuda()
            y = mech.cuda()[:, 1]  # Predicts whether it is flipped or not.
            pred = net(im)
            pred = (pred > 0.5).squeeze()
            correct += (pred == y).sum()
            total += pred.shape[0]
    return (correct/total).item()


N_EPOCHS = 100
if __name__ == '__main__':
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')

    train_set, val_set = get_datasets(raw_results, n_train=80)
    net = CNN()
    net.cuda()
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(net.parameters())

    for _ in range(N_EPOCHS):
        for policy_type, theta, im, y, mech in train_set:
            im = im.cuda()
            y = mech.cuda()[:, 1]  # Predicts whether it is flipped or not.

            optim.zero_grad()
            pred = net(im)
            loss = loss_fn(pred.squeeze(), y)
            loss.backward()
            optim.step()

        train_acc = evaluate(net, train_set)
        val_acc = evaluate(net, val_set)
        print('Training Accuracy:', train_acc)
        print('Val Accuracy:', val_acc)
