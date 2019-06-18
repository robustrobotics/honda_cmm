import torch

from torch import nn
from torch.autograd import Variable
from learning.set2set import Set2Set

# Module for residual/skip connections
class FCResBlock(nn.Module):
    def __init__(self, width, n, nonlinearity):
        """

        :param width:
        :param n:
        :param nonlinearity:
        """
        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.block = nn.ModuleList([nn.Linear(width, width) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)


# PRE-POOLING FOR STATISTIC NETWORK
class PrePool(nn.Module):
    """

    """

    def __init__(self, n_features, n_hidden, hidden_dim, nonlinearity):
        super(PrePool, self).__init__()
        self.n_features = n_features

        self.n_hidden = 1#n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_initial = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)
        self.fc_final = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        # reshape and initial affine
        e = x.view(-1, self.n_features)
        e = self.fc_initial(e)
        e = self.nonlinearity(e)

        # residual block
        e = self.fc_block(e)

        # final affine
        e = self.fc_final(e)

        return e


# POST POOLING FOR STATISTIC NETWORK
class PostPool(nn.Module):
    """

    """

    def __init__(self, n_hidden, hidden_dim, c_dim, nonlinearity):
        super(PostPool, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden,
                                   nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        e = self.fc_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        #e = e.view(-1, 1, 2 * self.c_dim)
        #e = self.bn_params(e)
        e = e.view(-1, 2 * self.c_dim)

        mean, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mean, logvar


# STATISTIC NETWORK q(c|D)
class StatisticNetwork(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, nonlinearity):
        super(StatisticNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity
        self.set2set = Set2Set(n_features, hidden_dim)

        # modules
        self.prepool = PrePool(self.n_features, 1,
                               self.hidden_dim, self.nonlinearity)
        self.postpool = PostPool(self.n_hidden, self.hidden_dim,
                                 self.c_dim, self.nonlinearity)

    def forward(self, x):
        #print('X', x.shape)
        e = self.prepool(x)
        #print('E1', e.shape)
        #e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        # hs = []
        # for ix in range(self.batch_size):
        #     h, _ = self.set2set(e[ix, :, :], timesteps=3)
        #     hs.append(h.squeeze().unsqueeze(0))
        # e = torch.cat(hs, dim=0)
        #e = self.set2set(x, timesteps=10)


        #print('E2', e.shape)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    def pool(self, e):
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(self.batch_size, self.hidden_dim)
        return e


# INFERENCE NETWORK q(z|x, z, c)
class InferenceNetwork(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(InferenceNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_down = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc_x = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        #self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, x, z, c):
        # combine x, z, and c
        # embed x
        ex = x.view(-1, self.n_features)
        ex = self.fc_x(ex)
        ex = ex.view(self.batch_size, self.sample_size, self.hidden_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(ex.size()).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ex)

        # sum and reshape
        #e = torch.cat([ex, ec], dim=2)
        #e = self.fc_down(e)
        e = ex + ec
        e = e.view(self.batch_size * self.sample_size, self.hidden_dim)
        e = self.nonlinearity(e)

        # residual blocks
        e = self.fc_block1(e)
        e = self.fc_block2(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        #e = e.view(-1, 1, 2 * self.z_dim)
        #e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()
        return mean, logvar


# LATENT DECODER p(z|z, c)
class LatentDecoder(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(LatentDecoder, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)
        self.fc_down = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, z, c):
        # combine z and c
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(self.batch_size, 1, self.hidden_dim).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ez)

        # sum and reshape
        e = torch.cat([ez, ec], dim=2)
        e = self.fc_down(e)
        e = e.view(-1, self.hidden_dim)
        e = self.nonlinearity(e)

        # residual blocks
        e = self.fc_block1(e)
        e = self.fc_block2(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# Observation Decoder p(x|z, c)
class ObservationDecoder(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features, n_hidden, hidden_dim, c_dim, n_stochastic, z_dim, nonlinearity):
        super(ObservationDecoder, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.n_stochastic = n_stochastic
        self.z_dim = z_dim

        self.nonlinearity = nonlinearity


        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.c_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden-1, nonlinearity=self.nonlinearity)
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.n_features)

        self.logvar = nn.Parameter(torch.randn(n_features))
        self.down_fc = nn.Linear(self.hidden_dim*2, self.hidden_dim)

    def forward(self, zs, c):
        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.sample_size, self.hidden_dim)

        ec = self.fc_c(c)
        ec = self.c_block(ec)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)

        #e = torch.cat([ezs, ec], dim=2)
        #e = self.down_fc(e)
        e = ec + ezs
        e = self.nonlinearity(e)
        e = e.view(-1, self.hidden_dim)

        e = self.fc_block(e)

        e = self.fc_params(e)

        mean, logvar = e[:, :self.n_features], e[:, self.n_features:]
        return mean, self.logvar.expand_as(mean)
