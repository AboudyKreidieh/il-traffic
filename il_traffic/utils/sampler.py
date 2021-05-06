"""Script containing the environment sampler method."""
import ray
import gym
from copy import deepcopy

from il_traffic.utils.flow_utils import create_env
from il_traffic.utils.flow_utils import get_base_env_params
from il_traffic.utils.flow_utils import get_rl_env_params
from il_traffic.utils.flow_utils import get_network_params


class Sampler(object):
    """Environment sampler object.

    Attributes
    ----------
    env_name : str
        the name of the environment
    render : bool
        whether to render the environment
    expert : int
        the expert policy used
    env_num : int
        the environment number. Used to handle situations when multiple
        parallel environments are being used.
    env : gym.Env
        the training environment
    """

    def __init__(self, env_name, render, expert, env_params, env_num):
        """Instantiate the sampler object.

        Parameters
        ----------
        env_name : str
            the name of the environment
        render : bool
            whether to render the environment
        expert : int
            the expert policy used
        env_params : dict
            dictionary of environment-specific parameters
        env_num : int
            the environment number. Used to handle situations when multiple
            parallel environments are being used.
        """
        self.env_name = env_name
        self.render = render
        self.expert = expert
        self.env_num = env_num

        # Create the environment.
        if env_name in ["highway", "i210"]:
            self.env = self._create_highway_i210_env(
                env_name=env_name,
                expert=expert,
                env_params=env_params,
                render=render and env_num == 0,
            )
        else:
            # for other gym-compatible environments
            self.env = gym.make(env_name)

        self._init_obs = self.env.reset()

    @staticmethod
    def _create_highway_i210_env(env_name, expert, env_params, render):
        """Return the flow environment for the highway and I-210.

        Parameters
        ----------
        env_name : str
            the name of the environment
        expert : int
            the expert policy used
        env_params : dict
            dictionary of environment-specific parameters
        render : bool
            whether to render the environment

        Returns
        -------
        gym.Env
            the environment
        """
        # Get the network parameters.
        network_params = get_network_params(
            inflow=2000,
            end_speed=5,
            penetration_rate=0.05,
        )

        # Get the environment parameters.
        environment_params = get_base_env_params(
            network_type=env_name,
            network_params=network_params,
            controller_type=expert,
            save_video=False,
            noise=0,
            verbose=False,
        )
        environment_params.update(get_rl_env_params(env_name))

        # Update the environment parameters based on the overriding values from
        # the algorithm class.
        if "obs_frames" in env_params:
            environment_params["obs_frames"] = env_params["obs_frames"]
        if "frame_skip" in env_params:
            environment_params["frame_skip"] = env_params["frame_skip"]
        if "full_history" in env_params:
            environment_params["full_history"] = env_params["full_history"]
        if "avg_speed" in env_params:
            environment_params["avg_speed"] = env_params["avg_speed"]

        # Create the environment.
        return create_env(
            network_type=env_name,
            network_params=network_params,
            environment_params=environment_params,
            render=render,
            training=True,
        )

    def get_init_obs(self):
        """Return the initial observation from the environment."""
        return self._init_obs.copy()

    def observation_space(self):
        """Return the environment's observation space."""
        return self.env.observation_space

    def action_space(self):
        """Return the environment's action space."""
        return self.env.action_space

    def horizon(self):
        """Return the environment's time horizon."""
        if hasattr(self.env, "horizon"):
            return self.env.horizon
        elif hasattr(self.env, "_max_episode_steps"):
            return self.env._max_episode_steps
        elif hasattr(self.env, "env_params"):
            # for Flow environments
            return self.env.env_params.horizon
        else:
            raise ValueError("Horizon attribute not found.")

    def collect_sample(self, obs, action):
        """Perform the sample collection operation over a single step.

        This method is responsible for executing a single step of the
        environment. This is perform a number of times in the _collect_samples
        method before training is executed. The data from the rollouts is
        stored in the policy's replay buffer(s).

        Parameters
        ----------
        obs : array_like
            the current observation
        action : array_like or None
            the action to be performed by the agent(s) within the environment.
            If set to None, the expert controls the actions.

        Returns
        -------
        dict
            information from the most recent environment update step,
            consisting of the following terms:

            * obs0 : the previous observation
            * obs1 : the most recent observation
            * action : the action performed by the agent(s)
            * reward : the reward from the most recent step
            * done : the done mask
            * env_num : the environment number
            * info : the info dict
            * expert_action : the action that would be performed by the expert
        """
        # Ensure that all AVs are in the dictionary to begin with.
        controlled_rl_ids = deepcopy(self.env.rl_ids)
        for veh_id in controlled_rl_ids:
            if veh_id not in self.env.av_controllers_dict.keys():
                controller = deepcopy(self.env._av_controller)
                controller.veh_id = veh_id
                self.env.av_controllers_dict[veh_id] = deepcopy(controller)

        # Query the expert for its desired action.
        expert_action = self.env.query_expert()

        # Execute the next action.
        next_obs, reward, done, info = self.env.step(action)

        # Visualize the current step.
        if self.render and self.env_num == 0:
            self.env.render()

        # If done, reset the environment and pass the initial observation.
        if done:
            next_obs = self.env.reset()

        return {
            "obs0": obs,
            "obs1": next_obs,
            "action": action,
            "reward": reward,
            "done": done,
            "env_num": self.env_num,
            "info": info,
            "expert_action": expert_action,
        }


@ray.remote
class RaySampler(Sampler):
    """Ray-compatible variant of the environment sampler object.

    Used to collect samples in parallel.
    """

    pass
