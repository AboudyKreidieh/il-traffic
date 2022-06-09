"""Tests for the features in scripts/."""
import unittest
import os
import shutil

import il_traffic.config as config
from il_traffic.scripts.simulate import parse_args as parse_simulate_args
from il_traffic.scripts.simulate import main as simulate
from il_traffic.scripts.imitate import parse_args as parse_imitate_args
from il_traffic.scripts.imitate import get_hyperparameters
from il_traffic.scripts.imitate import main as imitate
from il_traffic.scripts.evaluate import parse_args as parse_evaluate_args
from il_traffic.scripts.evaluate import main as evaluate


class TestSimulate(unittest.TestCase):
    """Tests for the features in scripts/simulate.py."""

    def test_parse_args(self):
        """Test the parse_options method.

        This is done for the following cases:

        1. default case
        2. custom case
        """
        # test case 1
        args = parse_simulate_args([])
        expected_args = {
            'controller_type': 1,
            'end_speed': 5.0,
            'gen_emission': False,
            'inflow': 2000,
            'network_type': 'i210',
            'noise': 0.0,
            'penetration_rate': 0.05,
            'render': False,
            'save_video': False,
            'use_warmup': False,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_simulate_args([
            "--network_type", "1",
            "--inflow", "4",
            "--end_speed", "5",
            "--penetration_rate", "6",
            "--controller_type", "7",
            "--render",
            "--gen_emission",
            "--noise", "8",
            "--save_video",
            "--use_warmup",
        ])
        expected_args = {
            "network_type": "1",
            "inflow": 4,
            "end_speed": 5,
            "penetration_rate": 6,
            "controller_type": 7,
            "render": True,
            "gen_emission": True,
            "use_warmup": True,
            'noise': 8.0,
            "save_video": True,
        }
        self.assertDictEqual(vars(args), expected_args)

    def test_main(self):
        """Validate the functionality of the main methods.

        This test performs the following tests:

        1. that the emission file is properly generated
        2. that the proper images are created
        """
        # shrinks the horizon to speedup tests
        os.environ["TEST_FLAG"] = "True"

        # Run the main method.
        simulate(["--network_type", "highway",
                  "--gen_emission",
                  "--use_warmup"])

        # test case 1
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/emission.csv"))

        # test case 2
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/mpg.csv"))
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/mpg.png"))
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/ts-0.png"))
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/avg-speed.png"))
        self.assertTrue(os.path.exists(
            "expert_data/highway/DownstreamController/2000-5/tt.json"))

        # Delete all files.
        if len(os.listdir("expert_data/highway/DownstreamController")) == 1:
            shutil.rmtree("expert_data")


class TestImitate(unittest.TestCase):
    """Tests for the features in scripts/imitate.py."""

    def test_parse_args(self):
        """Test the parse_options method.

        This is done for the following cases:

        1. default case
        2. custom case
        """
        # test case 1
        args = parse_imitate_args([])
        expected_args = {
            'n_training': 1,
            'env_name': 'i210',
            'expert': 1,
            'batch_size': 128,
            'buffer_size': 2000000,
            'num_rollouts': 10,
            'num_train_steps': 1000,
            'num_iterations': 200,
            'seed': 1,
            'env_params:obs_frames': 5,
            'env_params:frame_skip': 5,
            'model_params:layers': [32, 32, 32],
            'model_params:learning_rate': 0.001,
            'model_params:dropout': False,
            'model_params:l2_penalty': 0,
            'model_params:stochastic': False,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_imitate_args([
            '--n_training', '1',
            '--env_name', '2',
            '--expert', '3',
            '--batch_size', '5',
            '--buffer_size', '6',
            '--num_rollouts', '8',
            '--num_train_steps', '9',
            '--num_iterations', '10',
            '--seed', '12',
            '--env_params:obs_frames', '13',
            '--env_params:frame_skip', '14',
            '--env_params:avg_speed',
            '--model_params:layers', '15', '16', '17',
            '--model_params:learning_rate', '18',
            '--model_params:dropout',
            '--model_params:l2_penalty', '19',
            '--model_params:stochastic',
        ])
        expected_args = {
            'n_training': 1,
            'env_name': '2',
            'expert': 3,
            'batch_size': 5,
            'buffer_size': 6,
            'num_rollouts': 8,
            'num_train_steps': 9,
            'num_iterations': 10,
            'seed': 12,
            'env_params:obs_frames': 13,
            'env_params:frame_skip': 14,
            'model_params:layers': [15, 16, 17],
            'model_params:learning_rate': 18,
            'model_params:dropout': True,
            'model_params:l2_penalty': 19,
            'model_params:stochastic': True,
        }
        self.assertDictEqual(vars(args), expected_args)

    def test_get_hyperparameters(self):
        """Validate the functionality of the get_hyperparameters method."""
        hp = get_hyperparameters(parse_imitate_args([]), seed=1000)
        expected_hp = {
            'env_name': 'i210',
            'expert': 1,
            'render': False,
            'batch_size': 128,
            'buffer_size': 2000000,
            'num_rollouts': 10,
            'num_train_steps': 1000,
            'num_iterations': 200,
            'seed': 1000,
            'env_params': {
                'obs_frames': 5,
                'frame_skip': 5,
            },
            'model_params': {
                'layers': [32, 32, 32],
                'learning_rate': 0.001,
                'dropout': False,
                'l2_penalty': 0,
                'stochastic': False,
            }
        }
        self.assertDictEqual(hp, expected_hp)

    def test_main(self):
        """Validate the functionality of the main method."""
        # shrinks the horizon to speedup tests
        os.environ["TEST_FLAG"] = "True"

        # Run the main method.
        imitate(["--env_name", "highway",
                 "--num_iterations", "1",
                 "--num_rollouts", "1"])

        # Get the date/time.
        date_time = sorted(os.listdir("imitation_data/highway/expert=1"))[-1]

        # Check for the files.
        self.assertTrue(os.path.exists(
            "imitation_data/highway/expert=1/{}/hyperparameters.json".format(
                date_time)))
        self.assertTrue(os.path.exists(
            "imitation_data/highway/expert=1/{}/checkpoints".format(
                date_time)))
        self.assertTrue(os.path.exists(
            "imitation_data/highway/expert=1/{}/train.csv".format(date_time)))
        self.assertTrue(os.path.exists(
            "imitation_data/highway/expert=1/{}/tb_log".format(date_time)))

        # Delete all files.
        # if len(os.listdir("imitation_data/highway/expert=1")) == 1:
        #     shutil.rmtree("imitation_data")


class TestEvaluate(unittest.TestCase):
    """Tests for the features in scripts/evaluate.py."""

    def test_parse_args(self):
        """Test the parse_options method.

        This is done for the following cases:

        1. default case
        2. custom case
        """
        # test case 1
        args = parse_evaluate_args(["model_path"])
        expected_args = {
            'model_path': 'model_path',
            'ckpt_num': None,
            'inflow': 2000,
            'end_speed': 5.0,
            'penetration_rate': 0.05,
            'render': False,
            'use_warmup': False,
            'gen_emission': False,
            'save_video': False,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_evaluate_args([
            '--model_path', '1',
            '--ckpt_num', '2',
            '--inflow', '3',
            '--end_speed', '4',
            '--penetration_rate', '5',
            '--render',
            '--use_warmup',
            '--gen_emission',
            '--save_video',
        ])
        expected_args = {
            'model_path': '1',
            'ckpt_num': 2,
            'inflow': 3,
            'end_speed': 4,
            'penetration_rate': 5,
            'render': True,
            'use_warmup': True,
            'gen_emission': True,
            'save_video': True,
        }
        self.assertDictEqual(vars(args), expected_args)

    def test_main(self):
        """Validate the functionality of the main method."""
        # shrinks the horizon to speedup tests
        os.environ["TEST_FLAG"] = "True"

        model_path = os.path.join(
            config.PROJECT_PATH, "tests/test_files/model_highway")

        # Run the main method.
        evaluate([model_path, "--gen_emission", "--use_warmup"])

        # Check that the files were created.
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/emission.csv")))
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/mpg.csv")))
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/mpg.png")))
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/ts-0.png")))
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/avg-speed.png")))
        self.assertTrue(os.path.exists(os.path.join(
            model_path, "results/2000-5/tt.json")))

        # Delete all files.
        shutil.rmtree(os.path.join(model_path, "results"))


if __name__ == '__main__':
    unittest.main()
