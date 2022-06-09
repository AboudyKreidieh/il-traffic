"""Implementation of the DAgger algorithm.

See: https://arxiv.org/pdf/1011.0686.pdf
"""
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import deque

from il_traffic.algorithms.base import ILAlgorithm
from il_traffic.models.fcnet import FeedForwardModel
from il_traffic.models.fcnet import FEEDFORWARD_PARAMS
from il_traffic.utils.misc import dict_update

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


DAGGER_PARAMS = dict(
    # the model learning rate
    learning_rate=1e-3,
    # scale for the L2 regularization penalty
    l2_penalty=0.,
    # the number of elements in a batch when performing SGD
    batch_size=128,
    # the maximum number of samples to store
    buffer_size=2000000,
)


class DAgger(ILAlgorithm):
    """DAgger training algorithm.

    See: https://arxiv.org/pdf/1011.0686.pdf
    """

    def __init__(self, env, alg_params, model_params):
        """See parent class."""
        super(DAgger, self).__init__(
            env=env,
            alg_params=alg_params,
            model_params=model_params,
        )

        # Extract algorithm parameters.
        self.learning_rate = alg_params["learning_rate"]
        self.l2_penalty = alg_params["l2_penalty"]
        self.batch_size = alg_params["batch_size"]
        self.buffer_size = alg_params["buffer_size"]

        # Create an object to store samples.
        self.samples = {
            "obs": deque(maxlen=self.buffer_size),
            "expert_actions": deque(maxlen=self.buffer_size),
        }

        # Prepare the model params.
        params = {}
        params.update(FEEDFORWARD_PARAMS)
        params = dict_update(params, model_params or {})

        # Create the model.
        self.model = FeedForwardModel(
            ob_dim=self.env.observation_space.shape[0],
            ac_dim=self.env.action_space.shape[0],
            **params,
        )
        self.model.to(self.device)

        # Create an optimizer object.
        self._optimizer = torch.optim.Adam(
            self.model.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_penalty)

        # Create the loss function.
        self._loss_fn = nn.MSELoss()

    def save(self, log_dir, epoch):
        """See parent class."""
        name = os.path.join(
            log_dir, "checkpoints", "model-{}.tp".format(epoch))
        torch.save(self.model.state_dict(), name)

    def load(self, log_dir, epoch):
        """See parent class."""
        name = os.path.join(
            log_dir, "checkpoints", "model-{}.tp".format(epoch))
        self.model.load_state_dict(torch.load(name))

    def load_demos(self, demo_dir):
        """See parent class."""
        filenames = [x for x in os.listdir(demo_dir) if x.endswith(".pkl")]

        for ix, f_idx in enumerate(filenames):
            fname = os.path.join(demo_dir, f_idx)
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            # Extract demonstrations.
            s0, a0 = data[0]
            for i in range(1, len(data)):
                s1, a1 = data[i]
                a = a1 - a0
                self.samples["obs"].append(s0)
                self.samples["expert_actions"].append(a)
                s0 = s1
                a0 = a1

            print('{} Demo Trajectories Loaded. Total Experience={}'.format(
                ix + 1, len(self.samples["obs"])))

    def get_action(self, obs):
        """See parent class."""
        self.model.eval()

        action = []
        n_agents = len(obs)
        for i in range(n_agents):
            ac = self.model(FloatTensor(obs[i]))
            action.append(ac.detach().cpu().numpy())

        return action

    def add_sample(self, obs, action, expert_action, done):
        """See parent class."""
        n_agents = len(obs)
        for i in range(n_agents):
            self.samples["obs"].append(obs[i])
            self.samples["expert_actions"].append(expert_action[i])

    def update(self):
        """See parent class."""
        total_loss = []
        for _ in range(self.num_train_steps):
            # Sample a batch.
            batch_i = np.random.randint(
                0, len(self.samples["obs"]), size=self.batch_size)

            obs = FloatTensor(
                [self.samples["obs"][i] for i in batch_i], 0)
            target = FloatTensor(
                [self.samples["expert_actions"][i] for i in batch_i], 0)

            # Set the model to training mode.
            self.model.train()

            # Compute loss.
            loss = self._loss_fn(self.model(obs), target).mean()

            # Optimize the model.
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss.append(loss.item())

        return np.mean(total_loss)
