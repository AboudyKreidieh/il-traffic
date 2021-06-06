"""Tests for the features in utils/."""
import unittest
import os
import shutil
import pandas as pd
import csv
import random
import tensorflow as tf
import numpy as np

import il_traffic.config as config
from il_traffic.core.experts import IntelligentDriverModel
from il_traffic.core.experts import FollowerStopper
from il_traffic.core.experts import PISaturation
from il_traffic.core.experts import TimeHeadwayFollowerStopper
from il_traffic.utils.flow_utils import get_emission_path
from il_traffic.utils.flow_utils import get_network_params
from il_traffic.utils.flow_utils import get_base_env_params
from il_traffic.utils.flow_utils import get_rl_env_params
from il_traffic.utils.tf_util import make_session
from il_traffic.utils.tf_util import layer
from il_traffic.utils.tf_util import get_trainable_vars
from il_traffic.utils.visualize import process_emission
from il_traffic.utils.sampler import Sampler
from il_traffic.utils.misc import dict_update


class TestFlowUtils(unittest.TestCase):
    """Tests for the features in core/utils/flow_utils.py."""

    def test_get_emission_path(self):
        """Validate the functionality of the get_emission_path method.

        This method tests the following cases:

        1. controller_type = 0
        2. controller_type = 1
        3. controller_type = 2
        4. controller_type = 3
        5. network_type = "highway", inflow=10, end_speed=20
        6. network_type = "i210", inflow=1, end_speed=2
        """
        # test case 1
        self.assertEqual(
            get_emission_path(
                controller_type=0,
                network_type="highway",
                network_params={"inflow": 1, "end_speed": 2},
            ),
            "./expert_data/highway/IDM/1-2",
        )

        # test case 2
        self.assertEqual(
            get_emission_path(
                controller_type=1,
                network_type="highway",
                network_params={"inflow": 1, "end_speed": 2},
            ),
            "./expert_data/highway/FollowerStopper/1-2",
        )

        # test case 3
        self.assertEqual(
            get_emission_path(
                controller_type=2,
                network_type="highway",
                network_params={"inflow": 1, "end_speed": 2},
            ),
            "./expert_data/highway/PISaturation/1-2",
        )

        # test case 4
        self.assertEqual(
            get_emission_path(
                controller_type=3,
                network_type="highway",
                network_params={"inflow": 1, "end_speed": 2},
            ),
            "./expert_data/highway/TimeHeadwayFollowerStopper/1-2",
        )

        # test case 5
        self.assertEqual(
            get_emission_path(
                controller_type=0,
                network_type="highway",
                network_params={"inflow": 10, "end_speed": 20},
            ),
            "./expert_data/highway/IDM/10-20",
        )

        # test case 6
        self.assertEqual(
            get_emission_path(
                controller_type=0,
                network_type="i210",
                network_params={"inflow": 1, "end_speed": 2},
            ),
            "./expert_data/i210/IDM/1-2",
        )

    def test_get_network_params(self):
        """Validate the functionality of the get_network_params method."""
        # Create random variables for each parameter.
        inflow = random.random()
        end_speed = random.random()
        penetration_rate = random.random()

        # Check the output from the method.
        self.assertDictEqual(
            get_network_params(
                inflow=inflow,
                end_speed=end_speed,
                penetration_rate=penetration_rate,
            ),
            {"inflow": inflow,
             "end_speed": end_speed,
             "penetration_rate": penetration_rate}
        )

    def test_get_base_env_params(self):
        """Validate the functionality of the get_base_env_params method.

        This is done for the following cases:

        1. controller_type = 0
        2. controller_type = 1
        3. controller_type = 2
        4. controller_type = 3
        5. network_type = "highway"
        6. network_type = "i210"
        """
        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="highway",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=0,
            save_video=False,
            noise=0.2,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                a=1.3,
                b=2.0,
                noise=0.2,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=IntelligentDriverModel,
                control_range=[500, 2300],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="highway",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=1,
            save_video=False,
            noise=0,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                v_des=5.,
                max_accel=1,
                max_decel=1,
                noise=0,
                sim_step=0.4,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=FollowerStopper,
                control_range=[500, 2300],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="highway",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=2,
            save_video=False,
            noise=0,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                max_accel=1,
                max_decel=1,
                noise=0,
                meta_period=10,
                sim_step=0.4,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=PISaturation,
                control_range=[500, 2300],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

        # =================================================================== #
        # test case 4                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="highway",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=3,
            save_video=False,
            noise=0,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                max_accel=1,
                max_decel=1,
                noise=0,
                v_des=5.,
                sim_step=0.4,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=TimeHeadwayFollowerStopper,
                control_range=[500, 2300],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

        # =================================================================== #
        # test case 5                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="highway",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=1,
            save_video=False,
            noise=0,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                v_des=5,
                max_accel=1,
                max_decel=1,
                noise=0,
                sim_step=0.4,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=FollowerStopper,
                control_range=[500, 2300],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

        # =================================================================== #
        # test case 6                                                         #
        # =================================================================== #

        env_params = get_base_env_params(
            network_type="i210",
            network_params={
                "inflow": 2000,
                "end_speed": 5,
                "penetration_rate": 0.05,
            },
            controller_type=1,
            save_video=False,
            noise=0,
            verbose=True,
        )

        # Check the controller_params variable.
        self.assertDictEqual(
            env_params["controller_params"],
            dict(
                v_des=5,
                max_accel=1,
                max_decel=1,
                noise=0,
                sim_step=0.4,
            ),
        )
        del env_params["controller_params"]

        # Check the rest of the attributes.
        self.assertDictEqual(
            env_params,
            dict(
                controller_cls=FollowerStopper,
                control_range=[573.08, 2363.27],
                max_accel=1,
                max_decel=1,
                obs_frames=5,
                frame_skip=5,
                avg_speed=False,
                full_history=False,
                save_video=False,
            ),
        )

    def test_get_rl_env_params(self):
        """Validate the functionality of the get_rl_env_params method.

        This is done for the following cases:

        1.  highway
        2.  i210
        3. woops --> returns error
        """
        # test case 1
        self.assertDictEqual(
            get_rl_env_params("highway"),
            {"warmup_path": os.path.join(
                config.PROJECT_PATH, "warmup/highway"),
             "rl_penetration": 0.05},
        )

        # test case 2
        self.assertDictEqual(
            get_rl_env_params("i210"),
            {"warmup_path": os.path.join(
                config.PROJECT_PATH, "warmup/i210"),
             "rl_penetration": 0.05},
        )

        # test case 3
        self.assertRaises(
            AssertionError, get_rl_env_params, env_name="woops")

    def test_get_flow_params(self):
        """Validate the functionality of the get_flow_params method.

        This is done for the following cases:

        1. TODO
        """
        pass  # TODO


class TestMisc(unittest.TestCase):
    """Tests for the features in core/utils/misc.py."""

    def test_dict_update(self):
        """Validate the functionality of the dict_update method."""
        dict1 = {"hello": {"world": {"1": "foo"}}}
        dict2 = {"hello": {"world": {"2": "bar"}}}

        self.assertDictEqual(
            dict_update(dict1, dict2),
            {"hello": {"world": {"1": "foo", "2": "bar"}}}
        )


class TestSampler(unittest.TestCase):
    """Tests for the features in core/utils/sampler.py."""

    def test_init(self):
        """Validate the functionality of the __init__ method.

        This is done for the following cases:

        1. highway
        2. i210
        """
        # =================================================================== #
        #                             test case 1                             #
        # =================================================================== #

        # Create the sampler.
        sampler = Sampler(
            env_name="highway",
            render=False,
            expert=0,
            env_params={},
            env_num=0
        )

        # Check the wrapped environment.
        self.assertEqual(
            sampler.env.k.network.network.__class__.__name__, "HighwayNetwork")

        # Delete the sampler.
        del sampler

        # =================================================================== #
        #                             test case 2                             #
        # =================================================================== #

        # Create the sampler.
        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={},
            env_num=0
        )

        # Check the wrapped environment.
        self.assertEqual(
            sampler.env.k.network.network.__class__.__name__, "I210SubNetwork")

        # Delete the sampler.
        del sampler

    def test_get_init_obs(self):
        """Validate the functionality of the get_init_obs method."""
        # Set random seeds.
        random.seed(0)
        np.random.seed(0)

        # Create the sampler.
        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={},
            env_num=0
        )

        # Check the method.
        np.testing.assert_almost_equal(
            sampler.get_init_obs(),
            [[0.67440379, 0.68813417, 0.08272568, 0., 0., 0., 0.],
             [0.65403765, 0.6405847, 0.07847633, 0., 0., 0., 0.],
             [0.55033265, 0.57241793, 0.06852884, 0., 0., 0., 0.],
             [0.67242395, 0.66881029, 0.08011711, 0., 0., 0., 0.],
             [0.57355391, 0.58505743, 0.07091328, 0., 0., 0., 0.],
             [0.56028122, 0.55277353, 0.06863959, 0., 0., 0., 0.],
             [0.75747709, 0.79940943, 0.10132385, 0., 0., 0., 0.],
             [0.70922407, 0.74866493, 0.08715971, 0., 0., 0., 0.],
             [0.68881468, 0.7342854, 0.08938506, 0., 0., 0., 0.],
             [0.61602017, 0.55379662, 0.07281891, 0., 0., 0., 0.],
             [0.68930143, 0.63452829, 0.08423376, 0., 0., 0., 0.],
             [0.85034153, 0.80554593, 0.10424209, 0., 0., 0., 0.],
             [0.64709095, 0.59404376, 0.07486632, 0., 0., 0., 0.],
             [0.38948302, 0.43693076, 0.05184375, 0., 0., 0., 0.],
             [0.73156583, 0.81538934, 0.09701339, 0., 0., 0., 0.],
             [0.50219334, 0.54863601, 0.06918145, 0., 0., 0., 0.],
             [0.91591401, 1.04214463, 0.12260129, 0., 0., 0., 0.],
             [1.15461085, 0.92016595, 0.13154719, 0., 0., 0., 0.],
             [1.26521341, 1.3560602, 0.17844568, 0., 0., 0., 0.],
             [0.63744201, 0.63698249, 0.0817833, 0., 0., 0., 0.],
             [1.3107227, 1.39129068, 0.17418886, 0., 0., 0., 0.],
             [0.47987644, 0.32730042, 0.05462062, 0., 0., 0., 0.],
             [0.92549496, 1.02071994, 0.13251117, 0., 0., 0., 0.],
             [1.16896087, 1.31378323, 0.1898991, 0., 0., 0., 0.],
             [0.39458973, 0.2274378, 0.04537751, 0., 0., 0., 0.],
             [0.17717182, 0.21254028, 0.02958683, 0., 0., 0., 0.],
             [0.95581996, 1.09970734, 0.15581181, 0., 0., 0., 0.],
             [1.64386596, 1.72950468, 0.2483051, 0., 0., 0., 0.],
             [0.1307436, 0.12424366, 0.01902529, 0., 0., 0., 0.],
             [0.18737099, 0.12292322, 0.02432612, 0., 0., 0., 0.]]
        )

    def test_observation_space(self):
        """Validate the functionality of the observation_space method.

        This is done for the following cases:

        1. obs_frames=1, full_history=False, avg_speed=False,
        1. obs_frames=5, full_history=False, avg_speed=False,
        1. obs_frames=5, full_history=True,  avg_speed=False,
        1. obs_frames=1, full_history=False, avg_speed=True,
        1. obs_frames=5, full_history=False, avg_speed=True,
        1. obs_frames=5, full_history=True,  avg_speed=True,
        """
        # =================================================================== #
        #                             test case 1                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 1, "full_history": False, "avg_speed": False},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 3)

        del sampler

        # =================================================================== #
        #                             test case 2                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 5, "full_history": False, "avg_speed": False},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 7)

        del sampler

        # =================================================================== #
        #                             test case 3                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 5, "full_history": True, "avg_speed": False},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 15)

        del sampler

        # =================================================================== #
        #                             test case 4                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 1, "full_history": False, "avg_speed": True},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 4)

        del sampler

        # =================================================================== #
        #                             test case 5                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 5, "full_history": False, "avg_speed": True},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 8)

        del sampler

        # =================================================================== #
        #                             test case 6                             #
        # =================================================================== #

        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={
                "obs_frames": 5, "full_history": True, "avg_speed": True},
            env_num=0
        )

        self.assertEqual(sampler.observation_space().shape[0], 16)

        del sampler

    def test_action_space(self):
        """Validate the functionality of the action_space method."""
        # Create the sampler.
        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={},
            env_num=0
        )

        # Check the method.
        self.assertEqual(sampler.action_space().high, [1])
        self.assertEqual(sampler.action_space().low, [-1])

    def test_horizon(self):
        """Validate the functionality of the horizon method."""
        # Maintain the correct horizon for test.
        os.environ["TEST_FLAG"] = "False"

        # Create the sampler.
        sampler = Sampler(
            env_name="i210",
            render=False,
            expert=0,
            env_params={},
            env_num=0
        )

        # Check the method.
        self.assertEqual(sampler.horizon(), 1500)


class TestTfUtil(unittest.TestCase):
    """Tests for the features in core/utils/tf_util.py."""

    def test_make_session(self):
        """Validate the functionality of the make_session method.

        A session is created and we try to use it for a simple operation.
        """
        graph = tf.Graph()
        with graph.as_default():
            # Create the session.
            sess = make_session(num_cpu=3, graph=graph)

            # Run a simple test.
            a = tf.constant(1., dtype=tf.float32)
            b = tf.constant(2., dtype=tf.float32)
            self.assertAlmostEqual(sess.run(a+b), 3)

        # Close the session and clear the graph.
        tf.compat.v1.reset_default_graph()
        sess.close()

    def test_layer(self):
        """Check the functionality of the layer method.

        This method is tested for the following features:

        1. the number of outputs from the layer equals num_outputs
        2. the name is properly used
        3. the proper activation function applied if requested
        4. batch_norm is applied if requested
        5. dropout is applied if requested
        """
        # =================================================================== #
        # test case 1                                                         #
        # =================================================================== #

        # Choose a random number of outputs.
        num_outputs = random.randint(1, 10)

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test1',
            ),
            num_outputs=num_outputs,
            name="test1",
        )

        # Test the number of outputs.
        self.assertEqual(out_val.shape[-1], num_outputs)

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 2                                                         #
        # =================================================================== #

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test2',
            ),
            num_outputs=num_outputs,
            name="test2",
        )

        # Test the name matches what is expected.
        self.assertEqual(out_val.name, "test2/BiasAdd:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 3                                                         #
        # =================================================================== #

        # Create the layer.
        out_val = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test3',
            ),
            act_fun=tf.nn.relu,
            num_outputs=num_outputs,
            name="test3",
        )

        # Test that the name matches the activation function that was added.
        self.assertEqual(out_val.name, "Relu:0")

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 4                                                         #
        # =================================================================== #

        # Create the layer.
        _ = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test5',
            ),
            batch_norm=True,
            phase=tf.compat.v1.placeholder(
                tf.bool,
                name='phase5',
            ),
            dropout=False,
            rate=tf.compat.v1.placeholder(
                tf.float32,
                name='rate5',
            ),
            num_outputs=num_outputs,
            name="test5",
        )

        # Test that the batch_norm layer was added.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['bn_test5/beta:0',
             'bn_test5/gamma:0',
             'test5/bias:0',
             'test5/kernel:0']
        )

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

        # =================================================================== #
        # test case 5                                                         #
        # =================================================================== #

        # Create the layer.
        _ = layer(
            val=tf.compat.v1.placeholder(
                tf.float32,
                shape=(None, 1),
                name='input_test6',
            ),
            batch_norm=False,
            phase=tf.compat.v1.placeholder(
                tf.bool,
                name='phase6',
            ),
            dropout=True,
            rate=tf.compat.v1.placeholder(
                tf.float32,
                name='rate6',
            ),
            num_outputs=num_outputs,
            name="test6",
        )

        # Test the layer names.
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['test6/bias:0',
             'test6/kernel:0']
        )

        # Clear the graph.
        tf.compat.v1.reset_default_graph()


class TestVisualize(unittest.TestCase):
    """Tests for the features in core/utils/visualize.py."""

    def test_process_emission(self):
        """Validate the functionality of the process_emission method."""
        directory = os.path.join(
            config.PROJECT_PATH, "tests/test_files/process_emission")
        fp = os.path.join(directory, "emission.csv")
        if not os.path.exists(directory):
            os.mkdir(directory)

        # Create a simple emission file with more degrees of precision than
        # needed.
        d = {
            'time': [0.123456789],
            'id': [0.987654321],
            'type': [7],
            'speed': [7],
            'headway': [7],
            'target_accel_with_noise_with_failsafe': [7],
            'target_accel_no_noise_no_failsafe': [7],
            'target_accel_with_noise_no_failsafe': [7],
            'target_accel_no_noise_with_failsafe': [7],
            'realized_accel': [7],
            'edge_id': [7],
            'lane_number': [7],
            'relative_position': [7],
        }
        with open(fp, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(d.keys())
            writer.writerows(zip(*d.values()))

        # Run the process_emission method and read the precision of the new
        # emission file.
        process_emission(directory, verbose=True)
        df = pd.read_csv(fp)
        self.assertAlmostEqual(df.time[0], 0.123)
        self.assertAlmostEqual(df.id[0], 0.988)
        self.assertAlmostEqual(df.speed[0], 7)

        # Delete the created file.
        shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
