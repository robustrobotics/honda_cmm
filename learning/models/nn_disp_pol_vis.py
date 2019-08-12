import torch
import torch.nn as nn
from learning.modules.policy_encoder import PolicyEncoder
from learning.modules.image_encoder_spatialsoftmax import ImageEncoder as SpatialEncoder
from learning.modules.image_encoder import ImageEncoder as CNNEncoder
import torch.nn.functional as F


class DistanceRegressor(nn.Module):

    def __init__(self, policy_names, policy_dims, hdim, im_h, im_w, kernel_size=3, image_encoder='spatial'):
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
        :param image_encoder: Whether to use a CNN encoder or Spatial encoder.
        """
        super(DistanceRegressor, self).__init__()

        self.policy_modules = nn.ModuleDict()
        for name, dim in zip(policy_names, policy_dims):
            self.policy_modules[name] = PolicyEncoder(n_params=dim,
                                                      n_q=1,
                                                      n_layer=2,
                                                      n_hidden=hdim*2)

        if image_encoder == 'cnn':
            self.image_module = CNNEncoder(hdim=hdim,
                                           H=im_h,
                                           W=im_w,
                                           kernel_size=kernel_size)
        else:
            self.image_module = SpatialEncoder(hdim=hdim,
                                               kernel_size=7)

        self.fc1 = nn.Linear(hdim*4, hdim*4)
        self.fc2 = nn.Linear(hdim*4, hdim*2)
        self.fc3 = nn.Linear(hdim*2, hdim*2)
        self.fc4 = nn.Linear(hdim, hdim)
        self.fc5 = nn.Linear(hdim*2, 1)
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
        if policy_type == 0:
            policy_type = 'Prismatic'
        else:
            policy_type = 'Revolute'
        pol = self.policy_modules[policy_type].forward(theta, q)
        im = self.image_module(im)

        # x = pol*im
        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # x = x/x_norm
        x = torch.cat([pol, im], dim=1)
        # x = pol + im

        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        x = self.fc5(x)
        return x
