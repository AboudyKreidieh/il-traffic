"""Tests for the features in scripts/."""
import unittest

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
            'verbose': False,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_simulate_args([
            "--network_type", "1",
            "--inflow", "4",
            "--end_speed", "5",
            "--penetration_rate", "6",
            "--controller_type", "7",
            "--verbose",
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
            "verbose": True,
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
        # Run the main method.
        pass  # TODO

        # test case 1
        pass  # TODO

        # test case 2
        pass  # TODO


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
            'num_envs': 1,
            'render': False,
            'batch_size': 128,
            'buffer_size': 2000000,
            'prob_add': 1.0,
            'num_rollouts': 10,
            'num_train_steps': 1000,
            'num_iterations': 200,
            'initial_episodes': 20,
            'seed': 1,
            'env_params:obs_frames': 5,
            'env_params:frame_skip': 5,
            'env_params:full_history': False,
            'env_params:avg_speed': False,
            'model_params:layers': [32, 32, 32],
            'model_params:learning_rate': 0.001,
            'model_params:batch_norm': False,
            'model_params:dropout': False,
            'model_params:l2_penalty': 0,
            'model_params:stochastic': False,
            'model_params:num_ensembles': 1,
        }
        self.assertDictEqual(vars(args), expected_args)

        # test case 2
        args = parse_imitate_args([
            '--n_training', '1',
            '--env_name', '2',
            '--expert', '3',
            '--num_envs', '4',
            '--render',
            '--batch_size', '5',
            '--buffer_size', '6',
            '--prob_add', '7',
            '--num_rollouts', '8',
            '--num_train_steps', '9',
            '--num_iterations', '10',
            '--initial_episodes', '11',
            '--seed', '12',
            '--env_params:obs_frames', '13',
            '--env_params:frame_skip', '14',
            '--env_params:full_history',
            '--env_params:avg_speed',
            '--model_params:layers', '15', '16', '17',
            '--model_params:learning_rate', '18',
            '--model_params:batch_norm',
            '--model_params:dropout',
            '--model_params:l2_penalty', '19',
            '--model_params:stochastic',
            '--model_params:num_ensembles', '20',
        ])
        expected_args = {
            'n_training': 1,
            'env_name': '2',
            'expert': 3,
            'num_envs': 4,
            'render': True,
            'batch_size': 5,
            'buffer_size': 6,
            'prob_add': 7,
            'num_rollouts': 8,
            'num_train_steps': 9,
            'num_iterations': 10,
            'initial_episodes': 11,
            'seed': 12,
            'env_params:obs_frames': 13,
            'env_params:frame_skip': 14,
            'env_params:full_history': True,
            'env_params:avg_speed': True,
            'model_params:layers': [15, 16, 17],
            'model_params:learning_rate': 18,
            'model_params:batch_norm': True,
            'model_params:dropout': True,
            'model_params:l2_penalty': 19,
            'model_params:stochastic': True,
            'model_params:num_ensembles': 20,
        }
        self.assertDictEqual(vars(args), expected_args)

    def test_get_hyperparameters(self):
        """Validate the functionality of the get_hyperparameters method."""
        hp = get_hyperparameters(parse_imitate_args([]), seed=1000)
        expected_hp = {
            'env_name': 'i210',
            'expert': 1,
            'num_envs': 1,
            'render': False,
            'batch_size': 128,
            'buffer_size': 2000000,
            'prob_add': 1.0,
            'num_rollouts': 10,
            'num_train_steps': 1000,
            'num_iterations': 200,
            'initial_episodes': 20,
            'seed': 1000,
            'env_params': {
                'obs_frames': 5,
                'frame_skip': 5,
                'full_history': False,
                'avg_speed': False
            },
            'model_params': {
                'layers': [32, 32, 32],
                'learning_rate': 0.001,
                'batch_norm': False,
                'dropout': False,
                'l2_penalty': 0,
                'stochastic': False,
                'num_ensembles': 1
            }
        }
        self.assertDictEqual(hp, expected_hp)

    def test_main(self):
        """Validate the functionality of the main method.

        TODO.
        """
        pass  # TODO


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
        """Validate the functionality of the main method.

        TODO.
        """
        pass  # TODO


if __name__ == '__main__':
    unittest.main()
