"""Tests for the features in utils/."""
import unittest
import os
import shutil
import pandas as pd
import csv
import random
import tensorflow as tf

import il_traffic.config as config
from il_traffic import IntelligentDriverModel
from il_traffic import FollowerStopper
from il_traffic import PISaturation
from il_traffic import TimeHeadwayFollowerStopper
from il_traffic.utils.flow_utils import get_emission_path
from il_traffic.utils.flow_utils import get_network_params
from il_traffic.utils.flow_utils import get_base_env_params
from il_traffic.utils.flow_utils import get_rl_env_params
from il_traffic.utils.tf_util import make_session
from il_traffic.utils.tf_util import layer
from il_traffic.utils.tf_util import get_trainable_vars
from il_traffic.utils.visualize import process_emission


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
        2. TODO
        """
        pass  # TODO


class TestMisc(unittest.TestCase):
    """Tests for the features in core/utils/misc.py."""

    def test_dict_update(self):
        """Validate the functionality of the dict_update method."""
        pass  # TODO


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
        pass  # TODO

        # Check the attributes.
        pass  # TODO

        # Check the wrapped environment.
        pass  # TODO

        # Delete the sampler.
        pass  # TODO

        # =================================================================== #
        #                             test case 2                             #
        # =================================================================== #

        # Create the sampler.
        pass  # TODO

        # Check the attributes.
        pass  # TODO

        # Check the wrapped environment.
        pass  # TODO

        # Delete the sampler.
        pass  # TODO

    def test_get_init_obs(self):
        """Validate the functionality of the get_init_obs method."""
        # Create the sampler.
        pass  # TODO

        # Check the method.
        pass  # TODO

    def test_observation_space(self):
        """Validate the functionality of the observation_space method."""
        # Create the sampler.
        pass  # TODO

        # Check the method.
        pass  # TODO

    def test_action_space(self):
        """Validate the functionality of the action_space method."""
        # Create the sampler.
        pass  # TODO

        # Check the method.
        pass  # TODO

    def test_horizon(self):
        """Validate the functionality of the horizon method."""
        # Create the sampler.
        pass  # TODO

        # Check the method.
        pass  # TODO


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

    def test_get_global_position(self):
        """Validate the functionality of the get_global_position method.

        This is done for the following cases:

        1. highway
        2. i210
        """
        # =================================================================== #
        #                             test case 1                             #
        # =================================================================== #

        # Create a dataframe for the highway network.
        pass  # TODO

        # Create the global_position column.
        pass  # TODO

        # Check the output from the global_position column.
        pass  # TODO

        # =================================================================== #
        #                             test case 2                             #
        # =================================================================== #

        # Create a dataframe for the highway network.
        pass  # TODO

        # Create the global_position column.
        pass  # TODO

        # Check the output from the global_position column.
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
