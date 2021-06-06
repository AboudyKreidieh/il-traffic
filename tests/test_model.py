"""Tests for the features in core/model.py."""
import unittest
import tensorflow as tf

from il_traffic.core.model import FeedForwardModel
from il_traffic.utils.tf_util import get_trainable_vars


class TestFeedForwardModel(unittest.TestCase):
    """Tests for the features in FeedForwardModel."""

    def setUp(self):
        self.model_params = {
            "sess": tf.compat.v1.Session(),
            "ac_dim": 1,
            "ob_dim": 2,
            "layers": [128, 64, 32],
            "act_fun": tf.nn.relu,
            "learning_rate": 1e-5,
            "batch_norm": False,
            "dropout": False,
            "l2_penalty": False,
            "stochastic": False,
            "num_ensembles": 1,
        }

    def tearDown(self):
        self.model_params['sess'].close()
        del self.model_params

        # Clear the graph.
        tf.compat.v1.reset_default_graph()

    def test_init(self):
        """Check the functionality of the __init__() method.

        This method is tested for the following features:

        1. The proper structure graph was generated.
        2. All input placeholders and the output variable are correct.
        """
        # Create the model.
        model = FeedForwardModel(**self.model_params)

        # test case 1
        self.assertListEqual(
            sorted([var.name for var in get_trainable_vars()]),
            ['0/Model/fc0/bias:0',
             '0/Model/fc0/kernel:0',
             '0/Model/fc1/bias:0',
             '0/Model/fc1/kernel:0',
             '0/Model/fc2/bias:0',
             '0/Model/fc2/kernel:0',
             '0/Model/output/bias:0',
             '0/Model/output/kernel:0',
             'Variable:0',
             'Variable_1:0']
        )

        # test case 2
        self.assertEqual(model.obs_ph[0].shape[-1], model.ob_dim)
        self.assertEqual(model.expert_ph[0].shape[-1], model.ac_dim)
        self.assertEqual(model.output[0].shape[-1], model.ac_dim)


if __name__ == '__main__':
    unittest.main()
