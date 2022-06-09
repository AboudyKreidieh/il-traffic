"""Environment compatible with trajectory_training."""
import os
import random
import gym
import numpy as np
import trajectory.config as config
from gym.envs.registration import register

from il_traffic.experts.fs import DownstreamController
from il_traffic.experts.fs import TimeHeadwayFollowerStopper

# number of downstream edges to be sensed
N_DOWNSTREAM = 0
# simulation update steps per action assignment
SIMS_PER_STEP = 1
# the names of all valid trajectories for training purposes
FILENAMES = [
    '2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825',
    '2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917',
    '2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342',
    '2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_0_6438',
    '2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294',
    '2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116',
    '2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778',
    '2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467',
    '2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483',
    '2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050',
]


class TrajectoryEnv(gym.Env):
    """Environment compatible with trajectory_training.

    Separate environments are created for each trajectory, and at reset, one
    trajectory is sampled and advanced for the corresponding rollout.
    """

    def __init__(self):
        """Instantiate the environment."""
        # Construct all valid environments.
        self.all_envs = self._create_envs()

        # Choose a current environment.
        self.current_env = self._choose_env()

        # list of experts for each AV
        self.experts = []

        # a wrapper to convert speeds to accelerations
        self.fs = None

    @staticmethod
    def _create_envs():
        """Create a separate environment for each trajectory."""
        all_envs = []
        for i in range(len(FILENAMES)):
            env_config = {
                'horizon': 1000,
                'min_headway': 7.0,
                'max_headway': 120.0,
                'whole_trajectory': True,
                'discrete': False,
                'num_actions': 7,
                'use_fs': False,
                'augment_vf': False,
                'minimal_time_headway': 1.0,
                'include_idm_mpg': False,
                'num_concat_states': 1,
                'num_steps_per_sim': 1,
                'platoon': 'av',  # '(av human*25)*4',  # 'av',
                'av_controller': 'rl',
                'av_kwargs': '{}',
                'human_controller': 'idm',
                'human_kwargs': '{}',
                'fixed_traj_path': os.path.join(
                    config.PROJECT_PATH,
                    'dataset/data_v2_preprocessed_west/{}/'
                    'trajectory.csv'.format(FILENAMES[i])
                ),
                'lane_changing': False,
                # custom parameters
                "h_min": -1,
                "h_max": 150,
                "th_min": 1.5,
                "th_max": -1,
                "use_energy": True,
                "energy_steps": 5,
            }

            register(
                id="TrajectoryEnv-v{}".format(i),
                entry_point="trajectory.env.trajectory_env:TrajectoryEnv",
                kwargs={"config": env_config, "_simulate": True})

            # Make the gym environment.
            all_envs.append(gym.make("TrajectoryEnv-v{}".format(i)))

        return all_envs

    def _choose_env(self):
        """Return a random environment."""
        return random.sample(self.all_envs, 1)[0]

    def step(self, action):
        """Advance the simulation by one step.

        Parameters
        ----------
        action : list < float >
            the actions for each agent. If set to None, expert actions are
            applied.
        """
        n_agents = len(action)
        total_rew = [0. for _ in range(n_agents)]
        obs = None
        done = None
        info = None

        # Compute the expert actions.
        expert_vdes0 = []
        if self.current_env.avs[0].get_segments() is not None:
            expert_vdes0 = [self.experts[i].fs.v_des for i in range(n_agents)]
            expert_accel = [
                np.array([self.experts[i].get_action(
                    speed=av.speed,
                    headway=av.get_headway(),
                    lead_speed=av.get_leader_speed(),
                    avg_speed=np.asarray(av.get_avg_speed()[1]),
                    segments=np.asarray(av.get_segments()),
                    pos=av.pos,
                )])
                for i, av in enumerate(self.current_env.avs)
            ]
            expert_vdes1 = [self.experts[i].fs.v_des for i in range(n_agents)]

            expert_action = [
                np.array([expert_vdes1[i] - expert_vdes0[i]])
                for i in range(n_agents)]
        else:
            expert_accel = [np.array([0.]) for _ in range(n_agents)]
            expert_action = [np.array([0.]) for _ in range(n_agents)]

        accel = []
        if action[0] is not None:
            for i, av in enumerate(self.current_env.avs):
                # -1/+1 -> 0/40
                self.fs[i].v_des += float(action[i])
                self.fs[i].v_des = max(0., min(40., self.fs[i].v_des))
                accel.append(self.fs[i].get_action(
                    speed=av.speed,
                    headway=av.get_headway(),
                    lead_speed=av.get_leader_speed(),
                ))

        # Run for a few steps.
        for _ in range(SIMS_PER_STEP):
            obs, rew, done, info = self.current_env.step(
                accel if action[0] is not None else expert_accel)
            for i in range(len(rew)):
                total_rew[i] += rew[i] / SIMS_PER_STEP

            # Stop if simulation is terminated.
            if done:
                break

        # Add expert actions to the info dictionary.
        info["expert_action"] = expert_action

        return obs, total_rew, done, info

    def reset(self):
        """Reset the environment.

        A new trajectory is chosen and memory is cleared here.
        """
        # Clear memory from the prior environment. This is to deal with the
        # explosion in memory.
        if self.current_env.sim is not None:
            self.current_env.sim.data_by_vehicle.clear()

        # Choose a new environment.
        self.current_env = self._choose_env()

        # Reset the environment.
        obs = self.current_env.reset()

        # Create a DownstreamController expert for each AV.
        self.experts = [
            DownstreamController(0.1) for _ in range(len(obs))]

        # TODO
        self.fs = [
            TimeHeadwayFollowerStopper(30., 0.1) for _ in range(len(obs))]

        return obs

    def render(self, mode="human"):
        """See parent class."""
        pass

    @property
    def observation_space(self):
        """Return the observation space."""
        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(15 + 2 * N_DOWNSTREAM,))

    @property
    def action_space(self):
        """Return the action space."""
        return gym.spaces.Box(low=-1, high=1, shape=(1,))
