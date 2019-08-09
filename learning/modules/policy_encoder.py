"""
This file takes as input the current parameter policies and a desired
change in configuration. It outputs an encoded version of the policy.
"""
import torch
import torch.nn as nn

class PolicyEncoder(nn.Module):

    def __init__(self, n_params, n_q, n_layer, n_hidden):
        """
        A MLP that embeds specific policy parameters into a latent space.
        :param n_params: Dimension of the policy parameters.
        :param n_q: Dimension of the configuration variables.
        :param n_layer: How many fully connected layers to use.
        :param n_hidden: Size of the embedding.
        """
        super(PolicyEncoder, self).__init__()
        layers = [nn.Linear(n_params + n_q, n_hidden), nn.ReLU()]
        for _ in range(n_layer-1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers[:-1])

    def forward(self, theta, dq):
        """
        Embed a specific instantiation of the policy.
        :param theta: The policy parameters.
        :param dq: The change in configuration.
        :return: An embedding of size n_hidden.
        """
        x = torch.cat([theta, dq], dim=1)
        x = self.model.forward(x)
        return x
