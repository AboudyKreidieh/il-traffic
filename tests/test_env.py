"""Test for the objects in core/env.py."""
import unittest
import numpy as np
import os
import random
import tensorflow as tf
from copy import deepcopy

import il_traffic.config as config
from il_traffic import IntelligentDriverModel
from il_traffic.utils.flow_utils import create_env


class TestControllerEnv(unittest.TestCase):
    """Tests for the ControllerEnv object."""

    def setUp(self):
        self.highway_net_params = {
            "inflow": 2000,
            "end_speed": 5,
            "penetration_rate": 0.05,
        }
        self.highway_env_params = {
            "controller_cls": IntelligentDriverModel,
            "controller_params": {"a": 1.3, "b": 2.0, "noise": 0.2},
            "control_range": [500, 2300],
            "max_accel": 1,
            "max_decel": 1,
            "obs_frames": 5,
            "frame_skip": 5,
            "avg_speed": False,
            "full_history": 5,
            "warmup_path": None,
            "rl_penetration": 0.05,
            "save_video": False,
        }

    def test_init(self):
        """Validate the functionality of the __init__ method.

        This method performs the following tests:

        1. that the action space matches expected values
        2. that the observation space matches expected values
        """
        # Create the environment.
        env = create_env(
            network_type="highway",
            network_params=self.highway_net_params,
            environment_params=self.highway_env_params,
            render=False,
        )

        # test case 1
        test_space(
            gym_space=env.action_space,
            expected_size=1,
            expected_min=[-1],
            expected_max=[1],
        )

        # test case 2
        test_space(
            gym_space=env.observation_space,
            expected_size=15,
            expected_min=[-float("inf")] * 15,
            expected_max=[float("inf")] * 15,
        )

    def test_getter_attributes(self):
        """Validate the functionality of a variable of getter methods.

        This method resets a highway network from warmup and ensures that the
        following methods returned expected values:

        1. get_state
        2. compute_reward
        3. get_controlled_ids
        4. observed_ids
        """
        # Setup the seed value.
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

        env_params = deepcopy(self.highway_env_params)
        env_params["warmup_path"] = os.path.join(
            config.PROJECT_PATH, "tests/test_files/warmup_highway")

        # Create the environment.
        env = create_env(
            network_type="highway",
            network_params=self.highway_net_params,
            environment_params=env_params,
            render=False,
            warmup_steps=0,
        )

        # Reset the environment.
        env.reset()

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO

        # test case 3
        pass  # TODO

        # test case 4
        pass  # TODO

    def test_reset_highway(self):
        """Validate the functionality of the highway reset method.

        This method runs the following tests:

        1. that the warmup_description features are accurately filled.
        2. that the inflow and end_speed update appropriately
        """
        # Setup the seed value.
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

        env_params = deepcopy(self.highway_env_params)
        env_params["warmup_path"] = os.path.join(
            config.PROJECT_PATH, "tests/test_files/warmup_highway")

        # Create the environment.
        env = create_env(
            network_type="highway",
            network_params=self.highway_net_params,
            environment_params=env_params,
            render=False,
            warmup_steps=0,
        )

        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        self.assertListEqual(
            sorted(env._warmup_paths),
            ['0.xml', '1.xml', '2.xml', '3.xml'],
        )

        self.assertDictEqual(
            env._warmup_description,
            {'end_speed': [5.0, 5.0, 5.0, 5.0],
             'inflow': [1900.0, 1900.0, 1950.0, 1950.0],
             'xml_num': [0.0, 1.0, 2.0, 3.0]},
        )

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        self.assertEqual(
            env.network.net_params.inflows.get(),
            [{'name': 'idm_inflow_0',
              'vtype': 'human',
              'edge': 'highway_0',
              'departLane': 'free',
              'departSpeed': 24.1,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 1900},
             {'name': 'av_inflow_1',
              'vtype': 'av',
              'edge': 'highway_0',
              'departLane': 'free',
              'departSpeed': 24.1,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 100}]
        )
        self.assertEqual(env.k.network.get_max_speed("highway_end", 0), 5.0)

        env.reset()

        self.assertEqual(
            env.network.net_params.inflows.get(),
            [{'name': 'flow_0',
              'vtype': 'human',
              'edge': 'highway_0',
              'departLane': 'free',
              'departSpeed': 24.1,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 1852.5},
             {'name': 'flow_1',
              'vtype': 'av',
              'edge': 'highway_0',
              'departLane': 'free',
              'departSpeed': 24.1,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 97.5}]
        )
        self.assertEqual(env.k.network.get_max_speed("highway_end", 0), 5.0)

    def test_reset_i210(self):
        """Validate the functionality of the I-210 reset method.

        This method runs the following tests:

        1. that the warmup_description features are accurately filled.
        2. that the inflow and end_speed update appropriately
        """
        # Setup the seed value.
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.set_random_seed(1)

        env_params = deepcopy(self.highway_env_params)
        env_params["warmup_path"] = os.path.join(
            config.PROJECT_PATH, "tests/test_files/warmup_i210")

        # Create the environment.
        env = create_env(
            network_type="i210",
            network_params=self.highway_net_params,
            environment_params=env_params,
            render=False,
            warmup_steps=0,
        )

        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        self.assertListEqual(
            sorted(env._warmup_paths),
            ['0.xml', '1.xml', '2.xml', '3.xml'],
        )

        self.assertDictEqual(
            env._warmup_description,
            {'end_speed': [5.0, 5.0, 5.0, 5.0],
             'inflow': [1900.0, 1900.0, 1950.0, 1950.0],
             'xml_num': [0.0, 1.0, 2.0, 3.0]},
        )

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        self.assertEqual(
            env.network.net_params.inflows.get(),
            [{'name': 'human_inflow_0',
              'vtype': 'human',
              'edge': 'ghost0',
              'departLane': 'best',
              'departSpeed': 25.5,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 9500.0},
             {'name': 'av_inflow_1',
              'vtype': 'av',
              'edge': 'ghost0',
              'departLane': 'best',
              'departSpeed': 25.5,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 500.0}]
        )
        self.assertEqual(env.k.network.get_max_speed("119257908#3", 0), 5.0)

        env.reset()

        self.assertEqual(
            env.network.net_params.inflows.get(),
            [{'name': 'flow_0',
              'vtype': 'human',
              'edge': 'ghost0',
              'departLane': 'best',
              'departSpeed': 25.5,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 9262.5},
             {'name': 'flow_1',
              'vtype': 'av',
              'edge': 'ghost0',
              'departLane': 'best',
              'departSpeed': 25.5,
              'begin': 1,
              'end': 86400,
              'vehsPerHour': 487.5}]
        )
        self.assertEqual(env.k.network.get_max_speed("119257908#3", 0), 5.0)


# =========================================================================== #
#                               Auxiliary files                               #
# =========================================================================== #


def test_space(gym_space, expected_size, expected_min, expected_max):
    """Test that an action or observation space is the correct size and bounds.

    Parameters
    ----------
    gym_space : gym.spaces.Box
        gym space object to be tested
    expected_size : int
        expected size
    expected_min : float or array_like
        expected minimum value(s)
    expected_max : float or array_like
        expected maximum value(s)

    Returns
    -------
    bool
        True if the test passed, False otherwise
    """
    assert gym_space.shape[0] == expected_size, \
        "{}, {}".format(gym_space.shape[0], expected_size)
    np.testing.assert_almost_equal(gym_space.high, expected_max, decimal=4)
    np.testing.assert_almost_equal(gym_space.low, expected_min, decimal=4)


if __name__ == '__main__':
    unittest.main()
