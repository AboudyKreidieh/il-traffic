import os
import json
import torch
import torch.nn as nn
import gym
from gym import Env
from copy import deepcopy

from il_traffic.models.fcnet import FeedForwardModel

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class Expert(nn.Module):

    def __init__(self, state_dim, action_dim, train_config=None):
        super(Expert, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.train_config = train_config

        self.pi = FeedForwardModel(
            ob_dim=self.state_dim,
            ac_dim=self.action_dim,
            layers=[50, 50, 50],
            dropout=False,
            stochastic=True,
        )

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action


class GymEnv(Env):

    def __init__(self, env_name, device):
        self.env = gym.make(env_name)
        self.env.reset()

        expert_ckpt_path = "experts"
        expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

        with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
            expert_config = json.load(f)

        state_dim = len(self.env.observation_space.high)
        action_dim = self.env.action_space.shape[0]

        self.expert = Expert(
            state_dim,
            action_dim,
            **expert_config
        ).to(device)

        self.expert.pi.load_state_dict(
            torch.load(
                os.path.join(expert_ckpt_path, "policy.ckpt"),
                map_location=device
            )
        )

        self.obs = None

    def reset(self):
        ob = self.env.reset()
        self.obs = ob
        return [ob]

    def step(self, action):
        if action is not None:
            action = action[0]

        expert_action = self.expert.act(self.obs)

        ob, rew, done, info = self.env.step(action or expert_action)

        info["expert_action"] = [expert_action]

        self.obs = deepcopy(ob)

        return [ob], rew, done, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space
