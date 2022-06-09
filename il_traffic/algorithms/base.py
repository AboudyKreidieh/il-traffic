"""Base imitation learning algorithm."""
import torch.nn as nn


class ILAlgorithm(nn.Module):
    """Base imitation learning algorithm."""

    def __init__(self, env, alg_params, model_params):
        """Instantiate the base imitation algorithm.

        Parameters
        ----------
        env : TODO
            TODO
        alg_params : dict or None
            TODO
        model_params : dict or None
            dictionary of model-specific parameters. If set to None, default
            parameters are provided.
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
            TODO
        epoch : int
            the current training epoch
        """
        raise NotImplementedError

    def load(self, log_dir, epoch):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        log_dir : str
            TODO
        epoch : int
            the training epoch to load data from
        """
        raise NotImplementedError

    def load_demos(self, demo_dir):
        """Load initial expert demos.

        Parameters
        ----------
        demo_dir : str
            TODO
        """
        raise NotImplementedError

    def get_action(self, obs):
        """TODO.

        Parameters
        ----------
        obs : list of array_like
            TODO

        Returns
        -------
        list of array_like
            TODO
        """
        raise NotImplementedError

    def add_sample(self, obs, action, expert_action, done):
        """TODO.

        Parameters
        ----------
        obs : list of array_like
            TODO
        action : list of array_like
            TODO
        expert_action : list of array_like
            TODO
        done : bool
            TODO

        Returns
        -------
        TODO
            TODO
        """
        raise NotImplementedError

    def update(self):
        """Perform a policy optimization step.

        Returns
        -------
        TODO
            TODO
        """
        raise NotImplementedError
