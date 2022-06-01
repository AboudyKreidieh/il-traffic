"""Run the imitation operation."""
import sys
import os
import argparse
import json
from time import strftime

from flow.core.util import ensure_dir

from il_traffic.algorithms import DAgger
from il_traffic.models.fcnet import FeedForwardModel
from il_traffic.models.fcnet import FEEDFORWARD_PARAMS
from il_traffic.environments.flow_env import ADDITIONAL_ENV_PARAMS


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a imitation operation "
                    "on Flow environments.",
        epilog="python imitate.py --env_name i210")

    # optional input parameters
    parser.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')

    # algorithm-specific parameters
    parser.add_argument(
        '--env_name',
        type=str,
        default='i210',
        help='the environment to run. One of {"highway", "i210", "i24"}')
    parser.add_argument(
        '--expert',
        type=int,
        default=1,
        help='the type of expert, must be one of: '
             '0) Intelligent Driver Model, '
             '1) FollowerStopper, '
             '2) PISaturation, '
             '3) TimeHeadwayFollowerStopper')
    parser.add_argument(
        '--num_envs',
        type=int,
        default=1,
        help='number of environments used to run simulations in parallel. '
             'Each environment is run on a separate CPUS and uses the same '
             'policy as the rest. Must be less than or equal to num_rollouts')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Specifies whether to render the simulation during runtime.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='the number of elements in a batch when performing SGD')
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=2000000,
        help='the maximum number of samples to store')
    parser.add_argument(
        '--prob_add',
        type=float,
        default=1.0,
        help='the probability of adding any given sample to the buffer of '
             'training samples')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=10,
        help='number of rollouts to collect in between training iterations '
             'for the data aggregation procedure.')
    parser.add_argument(
        '--num_train_steps',
        type=int,
        default=1000,
        help='number of times a training operation is run in a given '
             'iteration of training')
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=200,
        help='number of training iterations')
    parser.add_argument(
        '--initial_episodes',
        type=int,
        default=20,
        help='initial number of episodes to collect from the expert policy')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='the random seed for numpy, tensorflow, and random')

    # environment-specific parameters
    parser.add_argument(
        '--env_params:obs_frames',
        type=int,
        default=ADDITIONAL_ENV_PARAMS["obs_frames"],
        help="number of observation frames to use. Additional frames are "
             "provided from previous time steps.")
    parser.add_argument(
        '--env_params:frame_skip',
        type=int,
        default=ADDITIONAL_ENV_PARAMS["frame_skip"],
        help="frames to ignore in between each delta observation")
    parser.add_argument(
        '--env_params:full_history',
        action="store_true",
        help="whether to use all observations from previous steps. If set to "
             "False, only the past speed is used.")
    parser.add_argument(
        '--env_params:avg_speed',
        action="store_true",
        help="whether to include the average speed of the leader vehicle in "
             "the observation")

    # model-specific parameters
    parser.add_argument(
        "--model_params:layers",
        type=int,
        nargs="+",
        default=FEEDFORWARD_PARAMS["layers"],
        help="the size of the neural network for the policy")
    parser.add_argument(
        '--model_params:learning_rate',
        type=float,
        default=FEEDFORWARD_PARAMS["learning_rate"],
        help="the model learning rate")
    parser.add_argument(
        "--model_params:batch_norm",
        action='store_true',
        help="whether to enable batch normalization")
    parser.add_argument(
        "--model_params:dropout",
        action='store_true',
        help="whether to enable dropout")
    parser.add_argument(
        '--model_params:l2_penalty',
        type=float,
        default=FEEDFORWARD_PARAMS["l2_penalty"],
        help="scale for the L2 regularization penalty")
    parser.add_argument(
        "--model_params:stochastic",
        action='store_true',
        help="whether the policy is stochastic or deterministic")
    parser.add_argument(
        '--model_params:num_ensembles',
        type=int,
        default=FEEDFORWARD_PARAMS["num_ensembles"],
        help="the number of ensemble models to use")

    return parser.parse_args(args)


def get_hyperparameters(flags, seed):
    """Return the hyperparameters of a training algorithm from the parser.

    Parameters
    ----------
    flags : argparse.Namespace
        the parser object
    seed : int
        the random seed for numpy, tensorflow, and random

    Returns
    -------
    dict
        the hyperparameters
    """
    hp = dict(
        env_name=flags.env_name,
        expert=flags.expert,
        num_envs=flags.num_envs,
        render=flags.render,
        batch_size=flags.batch_size,
        buffer_size=flags.buffer_size,
        prob_add=flags.prob_add,
        num_rollouts=flags.num_rollouts,
        num_train_steps=flags.num_train_steps,
        num_iterations=flags.num_iterations,
        initial_episodes=flags.initial_episodes,
        seed=seed,
        env_params=dict(
            obs_frames=getattr(flags, "env_params:obs_frames"),
            frame_skip=getattr(flags, "env_params:frame_skip"),
            full_history=getattr(flags, "env_params:full_history"),
            avg_speed=getattr(flags, "env_params:avg_speed"),
        ),
        model_params=dict(
            layers=getattr(flags, "model_params:layers"),
            learning_rate=getattr(flags, "model_params:learning_rate"),
            batch_norm=getattr(flags, "model_params:batch_norm"),
            dropout=getattr(flags, "model_params:dropout"),
            l2_penalty=getattr(flags, "model_params:l2_penalty"),
            stochastic=getattr(flags, "model_params:stochastic"),
            num_ensembles=getattr(flags, "model_params:num_ensembles"),
        ),
    )

    return hp


def main(args):
    """Run the imitation operation on a number of seeds."""
    # Parse input arguments.
    flags = parse_args(args)

    for i in range(flags.n_training):
        # value of the next seed
        seed = flags.seed + i

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join("imitation_data", "{}/expert={}/{}".format(
            flags.env_name, flags.expert, now))
        ensure_dir(dir_name)

        # Get the hyperparameters.
        hp = get_hyperparameters(flags, seed)

        # Add the date/time for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['model_cls'] = "FeedForwardModel"
        params_with_extra['date/time'] = now

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        # Create the algorithm object.
        alg = DAgger(model_cls=FeedForwardModel, log_dir=dir_name, **hp)

        # Start the training operation.
        alg.learn()


if __name__ == '__main__':
    main(sys.argv[1:])
