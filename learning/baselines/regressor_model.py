import argparse
import gpytorch
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.baselines.datagen import setup_data_loaders


class FeatureExtractor(nn.Module):
    def __init__(self, im_dim=3, im_h_dim=64, im_dropout=0.):
        super(FeatureExtractor, self).__init__()
        self.c, self.h, self.w = 3, 59, 58
        kernel_size, padding = 3, 1

        hidden_nonlin, final_nonlin = nn.ReLU, nn.Tanh
        self.im_net = nn.Sequential(*[nn.Conv2d(in_channels=im_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding), 
                                      hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                      hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                      hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                      nn.Conv2d(in_channels=im_h_dim, out_channels=im_h_dim, kernel_size=kernel_size, padding=padding),
                                      hidden_nonlin(), nn.Dropout(im_dropout), nn.MaxPool2d(kernel_size=2),
                                      nn.Flatten(), 
                                      nn.Linear(9*im_h_dim, im_h_dim), hidden_nonlin(), nn.Dropout(im_dropout),
                                      nn.Linear(im_h_dim, 1)])
        
    def forward(self, im):
        # Downsample
        bs, c, h, w = im.shape
        im = im[:, :, ::2, ::2]
        x = self.im_net(im)
        return x


def evaluate(extractor, dataset, use_cuda):
    """
    Return the RMSE of the GP predictions on the given dataset.
    """
    val_loss_fn = torch.nn.MSELoss()
    losses = []
    for im, y in dataset:
        if use_cuda:
            im = im.cuda()
            y = y.cuda()
        output = extractor(im)
        val_loss = val_loss_fn(output, y)
        losses.append(val_loss.item())
    return np.sqrt(np.mean(losses))
    
def train_exact_gp(args, train_set, val_set, use_cuda):
    # Setup the feature regressor.
    print('Extracting Features')
    extractor = FeatureExtractor(im_dropout=0.1)
    if use_cuda:
        extractor = extractor.cuda()


    # Setup optimizers and loss.
    optimizer = torch.optim.Adam([{'params': extractor.parameters(), 'lr': 1e-3}])
    loss_fn = torch.nn.MSELoss()

    # Training loop.
    best = 1000

    for tx in range(50000):
        for im, y in train_set:
            if use_cuda:
                im = im.cuda()
                y = y.cuda()

            optimizer.zero_grad()            
            output = extractor(im)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        if tx % 100 == 0:
            train_rmse = evaluate(extractor, train_set, use_cuda)
            val_rmse = evaluate(extractor, val_set, use_cuda)
            print('Train RMSE:', train_rmse)
            print('Val RMSE:', val_rmse)

                            
            # if tx % 50 == 0 and loss.item() < best:
            #     print('Saving to', args.save_path)
            #     torch.save((gp.state_dict(), train_xs, train_y, mu, std), args.save_path)
            #     best = loss.item()

    # gp.eval()
    # print('Val Predictions')
    # for ix in range(0, 20):#val_x.shape[0]):
    #     pred = likelihood(gp(val_x[ix:ix+1, :]))
    #     lower, upper = pred.confidence_region()
    #     lower = lower.cpu().detach().numpy()
    #     upper = upper.cpu().detach().numpy()
    #     true = val_y[ix].item()
        
    #     p = pred.mean.cpu().detach().numpy()[0]
    #     if true < lower or true > upper:
    #         print('INCORRECT:', pred.mean.cpu().detach().numpy(), 
    #             (lower[0], upper[0]), upper - lower,
    #             val_y[ix])
    #     elif upper[0]-lower[0] > 0.05:
    #         print('UNSURE:', pred.mean.cpu().detach().numpy(), 
    #             (lower[0], upper[0]), upper - lower,
    #             val_y[ix])
    #     else:
    #         print('CORRECT')

    # print('Train Predictions')
    # for ix in range(0, 20):
    #     pred = likelihood(gp(train_xs[ix:ix+1, :]))
    #     lower, upper = pred.confidence_region()
    #     lower = lower.cpu().detach().numpy()
    #     upper = upper.cpu().detach().numpy()
    #     print(pred.mean.cpu().detach().numpy(), 
    #           (lower[0], upper[0]), upper - lower,
    #           train_y[ix])


CUDA = False
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', default='')
    parser.add_argument('--L', required=True, type=int)
    args = parser.parse_args()
    print(args)

    #  Load dataset.
    with open('test.pkl', 'rb') as handle:
        raw_data = pickle.load(handle)[:args.L]
    train_set, val_set, test_set = setup_data_loaders(data=raw_data, 
                                                      batch_size=16)
    train_exact_gp(args, train_set, val_set, use_cuda=torch.cuda.is_available())


    
