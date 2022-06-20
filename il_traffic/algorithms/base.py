"""Base imitation learning algorithm."""
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import defaultdict


class ILAlgorithm(nn.Module):
    """Base imitation learning algorithm."""

    def __init__(self, env, alg_params, model_params):
        """Instantiate the base imitation algorithm.

        Parameters
        ----------
        env : Any
            the training environment
        alg_params : dict or None
            dictionary of algorithm-specific parameters
        model_params : dict or None
            dictionary of model-specific parameters
        """
        super(ILAlgorithm, self).__init__()

        self.env = env
        self.alg_params = alg_params
        self.model_params = model_params

    def save(self, log_dir, epoch):
        """Save a model's parameters is a specified path.

        Parameters
        ----------
        log_dir : str
            the directory to save checkpoint data within
        epoch : int
            the current training epoch
        """
        raise NotImplementedError

    def load(self, log_dir, epoch):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        log_dir : str
            the directory to load checkpoint data from
        epoch : int
            the training epoch to load data from
        """
        raise NotImplementedError

    def load_demos(self):
        """Load initial expert demos."""
        raise NotImplementedError

    def get_action(self, obs: list):
        """Return the desired actions by the policy.

        Parameters
        ----------
        obs : list
            list of agent observations

        Returns
        -------
        list of array_like
            list of agent actions
        """
        raise NotImplementedError

    def add_sample(self, obs: list, action: list, expert_action: list, done):
        """Store a sample within the algorithm class.

        Parameters
        ----------
        obs : list of array_like
            list of observations for each agent in the environment
        action : list of array_like
            list of actions by each agent in the environment
        expert_action : list of array_like
            list of actions assigned by an expert for each agent in the
             environment
        done : bool
            done mask
        """
        raise NotImplementedError

    def update(self):
        """Perform a policy optimization step.

        Returns
        -------
        float
            training loss
        """
        raise NotImplementedError

    def _get_i24_samples(self):
        """Return initial samples for the I-24 environment."""
        exp_obs = []
        exp_acts = []
        steps = 0
        info_data = defaultdict(list)
        for _ in range(9):
            done = False
            ep_rwds = []
            ob = self.env.reset()
            while not done:
                ob1, rwd, done, info = self.env.step(None)

                n_agents = len(ob)
                for i in range(n_agents):
                    exp_obs.append(ob[i])
                    exp_acts.append(info["expert_action"][i])
                ep_rwds.append(rwd)

                ob = deepcopy(ob1)
                steps += 1

            if done:
                for key in info.keys():
                    if key != "expert_action":
                        info_data[key].append(info[key])

        print("------------")
        for key in info_data.keys():
            print(key, np.mean(info_data[key]))
        print("------------")

        return exp_obs, exp_acts

    def _get_pendulum_samples(self):
        """Return initial samples for a Pendulum environment."""
        num_steps_per_iter = 2000
        exp_rwd_iter = []
        exp_obs = []
        exp_acts = []
        steps = 0
        while steps < num_steps_per_iter:
            done = False
            ep_rwds = []
            ob = self.env.reset()
            while not done and steps < num_steps_per_iter:
                ob1, rwd, done, info = self.env.step(None)

                n_agents = len(ob)
                for i in range(n_agents):
                    exp_obs.append(ob[i])
                    exp_acts.append(info["expert_action"][i])
                ep_rwds.append(rwd)

                ob = deepcopy(ob1)
                steps += 1

            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))

        exp_rwd_mean = np.mean(exp_rwd_iter)
        print("Expert Reward Mean: {}".format(exp_rwd_mean))

        return exp_obs, exp_acts
