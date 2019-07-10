import torch
import torch.nn as nn
from learning.policy_encoder import PolicyEncoder
from learning.image_encoder import ImageEncoder, PoseCNN, ImageEncoderOrig


class DistanceRegressor(nn.Module):

    def __init__(self, policy_names, policy_dims, hdim, im_h, im_w, kernel_size=3):
        """
        This module will estimate the distance the end-effector will move
        when executing various policies. It will have a separate module for
        each policy type.
        :param policy_names: The types of policies to instantiate modules for.
        :param policy_dims: The number of parameters associated with each policy.
        :param hdim: The number of hidden units to use in the neural net.
        :param im_h: Height of the input images.
        :param im_w: Width of the input images.
        :param kernel_size: Kernel size of the CNN.
        """
        super(DistanceRegressor, self).__init__()

        self.policy_modules = nn.ModuleDict()
        for name, dim in zip(policy_names, policy_dims):
            self.policy_modules[name] = PolicyEncoder(n_params=dim,
                                                      n_q=1,
                                                      n_layer=3,
                                                      n_hidden=hdim)

        self.image_module = ImageEncoder(hdim=hdim,
                                         H=im_h,
                                         W=im_w,
                                         kernel_size=kernel_size)

        self.fc1 = nn.Linear(hdim*2, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, 1)

        self.RELU = nn.ReLU()
        # SoftPLUS didn't work well... probably because our outputs are such small numbers.
        self.SOFTPLUS = nn.Softplus()

    def forward(self, policy_type, theta, q, im):
        """
        Call the distance regressor for a specific policy instantiation.
        :param policy_type: The name of the policy class being executed.
        :param theta: The policy parameters.
        :param q: How long the policy is executed for.
        :return:
        """
        im = self.image_module(im)
        x = self.policy_modules[policy_type].forward(theta, q)

        x = torch.cat([im, x], dim=1)
        x = self.fc3(self.RELU(self.fc2(self.RELU(self.fc1(x)))))
        return x
