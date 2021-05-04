"""Test for the objects in core/env.py."""
import unittest
import numpy as np
import os
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
            "control_range": None,
            "max_accel": 1,
            "max_decel": 1,
            "obs_frames": 5,
            "warmup_path": None,
            'rl_penetration': 0.05,
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
        np.test.assert_almost_equal(
            env.get_state(),
            [],
        )

        # test case 2
        np.test.assert_almost_equal(
            env.compute_reward(),
            [],
        )

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
            env._warmup_paths,
            []
        )

        self.assertDictEqual(
            env._warmup_description,
            {}
        )

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        self.assertEqual(env.net_params.inflow.get(), 2000)
        self.assertEqual(env.k.network.get_max_speed(""), 2000)

        env.reset()

        self.assertEqual(env.net_params.inflow.get(), 2000)
        self.assertEqual(env.k.network.get_max_speed(""), 2000)

    def test_reset_i210(self):
        """Validate the functionality of the I-210 reset method.

        This method runs the following tests:

        1. that the warmup_description features are accurately filled.
        2. that the inflow and end_speed update appropriately
        """
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
            env._warmup_paths,
            []
        )

        self.assertDictEqual(
            env._warmup_description,
            {}
        )

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        self.assertEqual(env.net_params.inflow.get(), 2000)
        self.assertEqual(env.k.network.get_max_speed(""), 2000)

        env.reset()

        self.assertEqual(env.net_params.inflow.get(), 2000)
        self.assertEqual(env.k.network.get_max_speed(""), 2000)

    def test_get_gallons(self):
        """Validate the functionality of the get_gallons method."""
        pass  # TODO


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
