import torch.nn as nn
from learning.modules.policy_encoder import PolicyEncoder


class DistanceRegressor(nn.Module):

    def __init__(self, policy_names, policy_dims, hdim):
        """
        This module will estimate the distance the end-effector will move
        when executing various policies. It will have a separate module for
        each policy type.
        :param policy_names: The types of policies to instantiate modules for.
        :param policy_dims: The number of parameters associated with each policy.
        :param hdim: The number of hidden units to use in the neural net.
        """
        super(DistanceRegressor, self).__init__()

        self.policy_modules = nn.ModuleDict()
        for name, dim in zip(policy_names, policy_dims):
            self.policy_modules[name] = PolicyEncoder(n_params=dim,
                                                      n_q=1,
                                                      n_layer=3,
                                                      n_hidden=hdim)

        self.fc1 = nn.Linear(hdim, hdim)
        self.fc2 = nn.Linear(hdim, 1)

        self.RELU = nn.ReLU()
        self.SOFTPLUS = nn.Softplus()

    def forward(self, policy_type, theta, q, im):
        """
        Call the distance regressor for a specific policy instantiation.
        :param policy_type: The name of the policy class being executed.
        :param theta: The policy parameters.
        :param q: How long the policy is executed for.
        :param im: Unused but kept for a consistent interface.
        :return:
        """
        x = self.policy_modules[policy_type].forward(theta, q)
        x = self.fc2(self.RELU(self.fc1(x)))
        return x
