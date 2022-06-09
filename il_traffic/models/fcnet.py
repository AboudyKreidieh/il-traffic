"""Script containing the FeedForwardModel object."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import MultivariateNormal


FEEDFORWARD_PARAMS = dict(
    # the size of the neural network for the policy
    layers=[32, 32, 32],
    # whether to enable dropout
    dropout=False,
    # whether the model should be stochastic or deterministic
    stochastic=False,
)


class FeedForwardModel(nn.Module):
    """Feedforward neural network model."""

    def __init__(self, ob_dim, ac_dim, layers, dropout, stochastic):
        """Instantiate the model.

        Parameters
        ----------
        ob_dim : int
            number of elements in the state space
        ac_dim : int
            number of elements in the action space
        dropout : bool
            whether to enable dropout
        stochastic : bool
            whether the policy is stochastic or deterministic
        """
        super(FeedForwardModel, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.layers = layers
        self.dropout = dropout
        self.stochastic = stochastic

        # Create the network.
        net_layers = []
        for i in range(len(layers)):
            in_dim = ob_dim if i == 0 else layers[i-1]
            out_dim = layers[i]
            net_layers.append(nn.Linear(in_dim, out_dim))
            if dropout:
                net_layers.append(nn.Dropout(0.5))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-1], ac_dim))

        self.net = nn.Sequential(*net_layers)

        print(self.net)

        # Initialize weights and biases.
        # for i, x in enumerate(self.net.modules()):
        #     if i == len(net_layers):
        #         nn.init.uniform_(x.weight, -3e-3, 3e-3)
        #     elif i >= 1 and (i-1) % (3 if dropout else 2) == 0:
        #         self.init_fanin(x.weight)

        # Create log_std for stochastic policies.
        if stochastic:
            self.log_std = nn.Parameter(torch.zeros(ac_dim))

    def forward(self, state):
        """Run a forward pass of the actor.

        Parameters
        ----------
        state : torch.Tensor
            the input state, (N,in_dim)

        Returns
        -------
        torch.Tensor
            deterministic action, (N,out_dim)
        """
        if self.stochastic:
            mean = self.net(state)
            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.ac_dim) * (std ** 2)
            distb = MultivariateNormal(mean, cov_mtx)
            return distb
        else:
            return self.net(state)

    @staticmethod
    def init_fanin(tensor):
        """Return fan-in initial parameters."""
        fanin = tensor.size(1)
        v = 1.0 / np.sqrt(fanin)
        init.uniform_(tensor, -v, v)
