"""Implementation of the GAIL algorithm.

See: https://arxiv.org/pdf/1703.08840.pdf
"""
import numpy as np
import os
import torch
from torch import FloatTensor

from il_traffic.algorithms.base import ILAlgorithm
from il_traffic.environments.gym_env import GymEnv
from il_traffic.environments.trajectory_env import TrajectoryEnv
from il_traffic.models.discriminator import Discriminator
from il_traffic.models.fcnet import FeedForwardModel
from il_traffic.utils.torch_utils import get_flat_grads
from il_traffic.utils.torch_utils import get_flat_params
from il_traffic.utils.torch_utils import set_params
from il_traffic.utils.torch_utils import conjugate_gradient
from il_traffic.utils.torch_utils import rescale_and_linesearch


GAIL_PARAMS = {
    # ======================================================================= #
    #                            Default parameters                           #
    # ======================================================================= #

    # TODO
    "lambda": 0.001,
    # GAE discount factor
    "gae_gamma": 0.99,
    # TODO
    "gae_lambda": 0.99,
    # TODO
    "epsilon": 0.01,
    # the Kullback-Leibler loss threshold
    "max_kl": 0.01,
    # the compute gradient dampening factor
    "cg_damping": 0.1,
    # whether to normalize the advantage
    "normalize_advantage": True,

    # ======================================================================= #
    #                           InfoGAIL parameters                           #
    # ======================================================================= #

    # TODO

    # ======================================================================= #
    #                      Directed InfoGAIL parameters                       #
    # ======================================================================= #

    # TODO
}


class GAIL(ILAlgorithm):
    """GAIL training algorithm and variants.

    See: https://arxiv.org/pdf/1703.08840.pdf
    """

    def __init__(self, env, alg_params, model_params):
        """See parent class."""
        super(GAIL, self).__init__(
            env=env,
            alg_params=alg_params,
            model_params=model_params,
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Extract algorithm parameters.
        self.lamda = alg_params["lambda"]
        self.gae_gamma = alg_params["gae_gamma"]
        self.gae_lambda = alg_params["gae_lambda"]
        self.epsilon = alg_params["epsilon"]
        self.max_kl = alg_params["max_kl"]
        self.cg_damping = alg_params["cg_damping"]
        self.normalize_advantage = alg_params["normalize_advantage"]

        # policy network
        self.pi = FeedForwardModel(
            ob_dim=state_dim,
            ac_dim=action_dim,
            layers=model_params["layers"],
            dropout=False,
            stochastic=True,
        )

        # value network
        self.v = FeedForwardModel(
            ob_dim=state_dim,
            ac_dim=1,
            layers=model_params["layers"],
            dropout=False,
            stochastic=False,
        )

        # discriminator
        self.d = Discriminator(
            ob_dim=state_dim,
            ac_dim=action_dim,
            layers=model_params["layers"],
        )

        # optimizer for the discriminator
        self.opt_d = torch.optim.Adam(self.d.parameters())

        # storage for expert data
        self.exp_obs = []
        self.exp_acts = []

        # storage for on-policy data
        self.obs = []
        self.acts = []
        self.rets = []
        self.advs = []
        self.gms = []

        # episodic statistics
        self._ep_obs = None
        self._ep_acts = None

    def save(self, log_dir, epoch):
        """See parent class."""
        ckpt_path = os.path.join(log_dir, "checkpoints")
        torch.save(
            self.pi.state_dict(),
            os.path.join(ckpt_path, f"policy-{epoch}.ckpt"))
        torch.save(
            self.v.state_dict(),
            os.path.join(ckpt_path, f"value-{epoch}.ckpt"))
        torch.save(
            self.d.state_dict(),
            os.path.join(ckpt_path, f"discriminator-{epoch}.ckpt"))

    def load(self, log_dir, epoch):
        """See parent class."""
        ckpt_path = os.path.join(log_dir, "checkpoints")
        self.pi.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"policy-{epoch}.ckpt")))
        self.v.load_state_dict(
            torch.load(os.path.join(ckpt_path, f"value-{epoch}.ckpt")))
        self.d.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, f"discriminator-{epoch}.ckpt")))

    def load_demos(self):
        """See parent class."""
        if isinstance(self.env, TrajectoryEnv):
            expert_obs, expert_acts = self._get_i24_samples()
        elif isinstance(self.env, GymEnv):
            expert_obs, expert_acts = self._get_pendulum_samples()
        else:
            raise NotImplementedError("Demos cannot be loaded.")

        self.exp_obs = FloatTensor(np.array(expert_obs))
        self.exp_acts = FloatTensor(np.array(expert_acts))

    def get_action(self, obs):
        """See parent class."""
        self.pi.eval()

        action = []
        n_agents = len(obs)
        for i in range(n_agents):
            distb = self.pi(FloatTensor(obs[i]))
            action.append(distb.sample().detach().cpu().numpy())

        return action

    def add_sample(self, obs, action, expert_action, done):
        """See parent class."""
        n_agents = len(obs)

        # Handle very first step.
        if self._ep_obs is None:
            self._ep_obs = [[] for _ in range(n_agents)]
            self._ep_acts = [[] for _ in range(n_agents)]

        # Add the agent's observations and actions.
        for i in range(n_agents):
            self._ep_obs[i].append(obs[i])
            self._ep_acts[i].append(action[i])

        if done:
            this_horizon = len(self._ep_obs[0])
            ep_obs = [FloatTensor(np.array(ob)) for ob in self._ep_obs]
            ep_acts = [FloatTensor(np.array(ac)) for ac in self._ep_acts]
            self.obs.extend(ep_obs)
            self.acts.extend(ep_acts)

            for i in range(n_agents):
                ep_gms = np.array([
                    self.gae_gamma ** t for t in range(this_horizon)])
                ep_lmbs = np.array([
                    self.gae_lambda ** t for t in range(this_horizon)])

                ep_costs = -torch.log(
                    self.d(ep_obs[i], ep_acts[i])).squeeze().detach().numpy()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = np.cumsum(ep_disc_costs[::-1])[::-1]
                ep_rets = FloatTensor(ep_disc_rets / ep_gms)

                self.rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs[i]).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs[i])[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = FloatTensor(ep_costs).unsqueeze(-1) \
                    + self.gae_gamma * next_vals - curr_vals

                ep_advs = FloatTensor([
                    (FloatTensor(ep_gms * ep_lmbs)[:this_horizon - j].
                     unsqueeze(-1)
                     * ep_deltas[j:]).sum() for j in range(this_horizon)])
                self.advs.append(ep_advs)

                self.gms.append(FloatTensor(ep_gms))

            self._ep_obs = [[] for _ in range(n_agents)]
            self._ep_acts = [[] for _ in range(n_agents)]

    def update(self):
        """See parent class."""
        obs = torch.cat(self.obs)
        acts = torch.cat(self.acts)
        rets = torch.cat(self.rets)
        advs = torch.cat(self.advs)
        gms = torch.cat(self.gms)

        if self.normalize_advantage:
            advs = (advs - advs.mean()) / advs.std()

        # =================================================================== #
        #                    Update discriminator network.                    #
        # =================================================================== #

        self.d.train()
        exp_scores = self.d.get_logits(self.exp_obs, self.exp_acts)
        nov_scores = self.d.get_logits(obs, acts)

        self.opt_d.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            exp_scores, torch.zeros_like(exp_scores)) \
            + torch.nn.functional.binary_cross_entropy_with_logits(
            nov_scores, torch.ones_like(nov_scores))
        loss.backward()
        self.opt_d.step()

        # =================================================================== #
        #                        Update value network.                        #
        # =================================================================== #

        self.v.train()
        old_params = get_flat_params(self.v).detach()
        old_v = self.v(obs).detach()

        def constraint():
            return ((old_v - self.v(obs)) ** 2).mean()

        grad_diff = get_flat_grads(constraint(), self.v)

        def Hv(v):
            hessian = get_flat_grads(
                torch.dot(grad_diff, v), self.v).detach()
            return hessian

        g = get_flat_grads(
            (-(self.v(obs).squeeze() - rets) ** 2).mean(), self.v).detach()
        s = conjugate_gradient(Hv, g).detach()

        Hs = Hv(s).detach()
        alpha = torch.sqrt(2 * self.epsilon / torch.dot(s, Hs))

        new_params = old_params + alpha * s

        set_params(self.v, new_params)

        # =================================================================== #
        #                        Update policy network.                       #
        # =================================================================== #

        self.pi.train()
        old_params = get_flat_params(self.pi).detach()
        old_distb = self.pi(obs)

        def L():
            distb = self.pi(obs)

            return (advs * torch.exp(
                distb.log_prob(acts) - old_distb.log_prob(acts).detach()
            )).mean()

        def kld():
            distb = self.pi(obs)
            old_mean = old_distb.mean.detach()
            old_cov = old_distb.covariance_matrix.sum(-1).detach()
            mean = distb.mean
            cov = distb.covariance_matrix.sum(-1)

            return 0.5 * (
                (old_cov / cov).sum(-1)
                + (((old_mean - mean) ** 2) / cov).sum(-1)
                - self.env.action_space.shape[0]
                + torch.log(cov).sum(-1)
                - torch.log(old_cov).sum(-1)).mean()

        grad_kld_old_param = get_flat_grads(kld(), self.pi)

        def Hv(v):
            hessian = get_flat_grads(
                torch.dot(grad_kld_old_param, v), self.pi).detach()
            return hessian + self.cg_damping * v

        g = get_flat_grads(L(), self.pi).detach()

        s = conjugate_gradient(Hv, g).detach()
        Hs = Hv(s).detach()

        new_params = rescale_and_linesearch(
            g, s, Hs, self.max_kl, L, kld, old_params, self.pi)

        disc_causal_entropy = (-gms * self.pi(obs).log_prob(acts)).mean()
        grad_disc_causal_entropy = get_flat_grads(
            disc_causal_entropy, self.pi)
        new_params += self.lamda * grad_disc_causal_entropy

        set_params(self.pi, new_params)

        # =================================================================== #
        #                           Reset storage.                            #
        # =================================================================== #

        self.obs = []
        self.acts = []
        self.rets = []
        self.advs = []
        self.gms = []

        return -1001  # error message for now
