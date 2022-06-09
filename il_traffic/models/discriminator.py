"""Discriminator class."""
import torch
import torch.nn as nn


DISCRIMINATOR_PARAMS = dict(
    # the size of the neural network for the policy
    layers=[32, 32, 32],
)


class Discriminator(nn.Module):
    """Discriminator class."""

    def __init__(self, ob_dim, ac_dim, layers):
        """Instantiate the model.

        Parameters
        ----------
        ob_dim : int
            number of elements in the state space
        ac_dim : int
            number of elements in the action space
        """
        super(Discriminator, self).__init__()

        # Create the network.
        net_layers = []
        for i in range(len(layers)):
            in_dim = (ob_dim + ac_dim) if i == 0 else layers[i-1]
            out_dim = layers[i]
            net_layers.append(nn.Linear(in_dim, out_dim))
            net_layers.append(nn.Tanh())
        net_layers.append(nn.Linear(layers[-1], 1))

        self.net = nn.Sequential(*net_layers)
        print(self.net)

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        sa = torch.cat([states, actions], dim=-1)
        return self.net(sa)
