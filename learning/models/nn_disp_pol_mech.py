import torch
import torch.nn as nn
from learning.modules.policy_encoder import PolicyEncoder
from learning.modules.image_encoder import ImageEncoder
import torch.nn.functional as F


class DistanceRegressor(nn.Module):

    def __init__(self, policy_names, policy_dims, mech_dims, hdim):
        """
        This module will estimate the distance the end-effector will move
        when executing various policies. It will have a separate module for
        each policy type.
        :param policy_names: The types of policies to instantiate modules for.
        :param policy_dims: The number of parameters associated with each policy.
        :param mech_dims: The number of mechanism specific parameters.
        :param hdim: The number of hidden units to use in the neural net.
        """
        super(DistanceRegressor, self).__init__()

        self.policy_modules = nn.ModuleDict()
        for name, dim in zip(policy_names, policy_dims):
            self.policy_modules[name] = PolicyEncoder(n_params=dim,
                                                      n_layer=2,
                                                      n_hidden=hdim)

        self.mech1 = nn.Linear(mech_dims, hdim)
        self.mech2 = nn.Linear(hdim, hdim)

        self.fc1 = nn.Linear(hdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc4 = nn.Linear(hdim, hdim)
        self.fc5 = nn.Linear(hdim, 1)
        # SoftPLUS didn't work well... probably because our outputs are such small numbers.
        self.SOFTPLUS = nn.Softplus()

    def forward(self, policy_type, theta, mech):
        """
        Call the distance regressor for a specific policy instantiation.
        :param policy_type: The name of the policy class being executed.
        :param theta: The policy parameters.
        :return:
        """
        if policy_type.item() == 0:
            policy_type = 'Prismatic'
        else:
            policy_type = 'Revolute'
        pol = self.policy_modules[policy_type].forward(theta)
        mech = F.relu(self.mech2(F.relu(self.mech1(mech))))

        x = pol*mech
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x/x_norm

        x = F.relu(self.fc1(x))
        x = self.fc5(x)
        return x
