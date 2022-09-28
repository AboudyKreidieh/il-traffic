"""Run the imitation operation."""
import sys
import os
import argparse
import json
import random
import numpy as np
import csv
import time
import torch
from copy import deepcopy
from collections import defaultdict
from time import strftime

import il_traffic.config as config
from il_traffic.algorithms import DAgger
from il_traffic.algorithms import GAIL
from il_traffic.algorithms.dagger import DAGGER_PARAMS
from il_traffic.algorithms.gail import GAIL_PARAMS
from il_traffic.models.fcnet import FEEDFORWARD_PARAMS
from il_traffic.utils.misc import ensure_dir


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
        epilog="python imitate.py --env_name i24")

    # optional input parameters
    parser.add_argument(
        '--n_training', type=int, default=1,
        help='Number of training operations to perform. Each training '
             'operation is performed on a new seed. Defaults to 1.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=9,
        help='number of rollouts to collect in between training iterations '
             'for the data aggregation procedure.')
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=200,
        help='number of training iterations')
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='the random seed for numpy, pytorch, and random')

    # ======================================================================= #
    #                     Environment-specific parameters                     #
    # ======================================================================= #

    parser.add_argument(
        '--env_name',
        type=str,
        default='i24',
        help='the environment to run. One of {"bottleneck", "i24"}')

    # ======================================================================= #
    #                      Algorithm-specific parameters                      #
    # ======================================================================= #

    parser.add_argument(
        '--alg_cls',
        type=str,
        default="DAgger",
        help="the algorithm class to use (in string format). One of "
             "{\"DAgger\", \"GAIL\"}")

    # DAgger
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=DAGGER_PARAMS["learning_rate"],
        help="the model learning rate")
    parser.add_argument(
        '--l2_penalty',
        type=float,
        default=DAGGER_PARAMS["l2_penalty"],
        help="scale for the L2 regularization penalty")
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=DAGGER_PARAMS["buffer_size"],
        help='the maximum number of samples to store')
    parser.add_argument(
        '--num_train_steps',
        type=int,
        default=DAGGER_PARAMS["num_train_steps"],
        help='number of training operations in a given iteration')

    # GAIL
    parser.add_argument(
        '--lambda',
        type=float,
        default=GAIL_PARAMS["lambda"],
        help="TODO")
    parser.add_argument(
        '--gae_gamma',
        type=float,
        default=GAIL_PARAMS["gae_gamma"],
        help="GAE discount factor")
    parser.add_argument(
        '--gae_lambda',
        type=float,
        default=GAIL_PARAMS["gae_lambda"],
        help="TODO")
    parser.add_argument(
        '--epsilon',
        type=float,
        default=GAIL_PARAMS["epsilon"],
        help="TODO")
    parser.add_argument(
        '--max_kl',
        type=float,
        default=GAIL_PARAMS["max_kl"],
        help="the Kullback-Leibler loss threshold")
    parser.add_argument(
        '--cg_damping',
        type=float,
        default=GAIL_PARAMS["cg_damping"],
        help="the compute gradient dampening factor")
    parser.add_argument(
        "--normalize_advantage",
        action='store_true',
        help="whether to normalize the advantage")

    # ======================================================================= #
    #                        Model-specific parameters                        #
    # ======================================================================= #

    # FeedForwardModel
    parser.add_argument(
        "--model_params:layers",
        type=int,
        nargs="+",
        default=FEEDFORWARD_PARAMS["layers"],
        help="the size of the neural network for the policy")
    parser.add_argument(
        "--model_params:dropout",
        action='store_true',
        help="whether to enable dropout")
    parser.add_argument(
        "--model_params:stochastic",
        action='store_true',
        help="whether the policy is stochastic or deterministic")

    return parser.parse_args(args)


class Trainer(object):
    """Base imitation learning algorithm object.

    This class is used to define most common operations between different
    imitation learning procedures. Models are trained via the `train` method,
    which is defined within individual subclasses.
    """

    def __init__(self,
                 env_name,
                 alg_cls,
                 alg_params,
                 model_params,
                 num_rollouts,
                 num_iterations,
                 log_dir):
        """Instantiate the base imitation algorithm.

        Parameters
        ----------
        env_name : str
            the name of the training environment
        alg_cls : str
            the algorithm class to use. One of {"DAgger", "GAIL"}
        alg_params : dict
            dictionary of algorithm-specific parameters
        model_params : dict
            dictionary of model-specific parameters
        num_rollouts : int
            number of rollouts to collect in between training iterations for
            the data aggregation procedure.
        num_iterations : int
            number of training iterations
        log_dir : str
            the directory where the training statistics and tensorboard log
            should be stored
        """
        self.num_rollouts = num_rollouts
        self.num_iterations = num_iterations
        self.log_dir = log_dir

        # a few initializations
        self.info_at_done = defaultdict(list)
        self.total_steps = 0
        self._info_keys = None

        # Create the torch device object.
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Create the environment.
        if env_name == "i24":
            from il_traffic.environments.trajectory import TrajectoryEnv
            self.env = TrajectoryEnv(n_vehicles=25, av_penetration=0.04)
        elif env_name == "bottleneck":
            from il_traffic.environments.bottleneck import BottleneckEnv
            self.env = BottleneckEnv(n_vehicles=1400, av_penetration=0.04)
        elif env_name == "Pendulum-v0":
            from il_traffic.environments.gym_env import GymEnv
            self.env = GymEnv(env_name, self.device)
            self.env.reset()
        else:
            raise NotImplementedError(
                "Unable to generate environment: {}".format(env_name))

        if alg_cls == "DAgger":
            alg_cls = DAgger
        elif alg_cls == "GAIL":
            alg_cls = GAIL
        else:
            raise ValueError("Unknown algorithm: {}".format(alg_cls))

        # Create the algorithm object.
        self.alg = alg_cls(self.env, alg_params, model_params).to(self.device)

        # Generate initial expert observations and actions.
        self.alg.load_demos()

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(self.log_dir)
        ensure_dir(os.path.join(self.log_dir, "checkpoints"))

        # file path for training and evaluation results
        self._train_filepath = os.path.join(self.log_dir, "train.csv")

    def learn(self):
        """Perform the complete training operation."""
        start_time = time.time()

        for itr in range(self.num_iterations):
            print("Iteration {}:\n".format(itr))

            # Simulation steps.
            print("\n- Generating samples from model.")
            action_lst, rew_lst = self.generate_rollouts()

            # Optimization steps.
            loss = self.alg.update()

            # Log and print results to csv.
            self.log_results(start_time, itr, loss, action_lst, rew_lst)

            # Save a checkpoint of the model.
            self.alg.save(log_dir=self.log_dir, epoch=itr)

    def generate_rollouts(self):
        """Create a number of rollouts and collect expert data.

        Returns
        -------
        list
            the list of actions performed by the agent
        list
            the list of cumulative episodic rewards
        """
        action_lst = []
        rew_lst = []
        for _ in range(self.num_rollouts):
            # Reset the environment.
            s = self.env.reset()

            done = False
            info = {}
            ep_rew = []
            while not done:
                self.total_steps += 1

                with torch.no_grad():
                    # Compute the action to be performed.
                    ac = self.alg.get_action(s)
                    action_lst.extend(ac)

                    # Run environment step.
                    s2, r, done, info = self.env.step(ac)
                    ep_rew.append(r)

                    # Store sample.  TODO
                    self.alg.add_sample(
                        obs=s,
                        action=ac,
                        expert_action=info["expert_action"],
                        done=done,
                    )

                    s = deepcopy(s2)

            # Some bookkeeping.
            rew_lst.append(sum(ep_rew))
            for key in info.keys():
                self.info_at_done[key].append(info[key])

        return action_lst, rew_lst

    def log_results(self, start_time, train_itr, loss, action_lst, rew_lst):
        """Log training statistics.

        Parameters
        ----------
        start_time : float
            the time when training began. This is used to print the total
            training time.
        train_itr : int
            the training iteration
        loss : float
            average imitation loss for the most recent epoch
        action_lst : list
            the list of actions performed by the agent
        rew_lst : list of float
            the list of cumulative episodic rewards
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            "rollout/imitation_loss": loss,
            "rollout/action_mean": np.mean(action_lst),
            "rollout/action_std": np.std(action_lst),
            "rollout/reward": np.mean(rew_lst),
            "total/epochs": train_itr + 1,
            "total/episodes": (train_itr + 1) * self.num_rollouts,
            "total/steps": self.total_steps,
            "total/duration": duration,
            "total/steps_per_second": self.total_steps / duration,
        }

        # Information passed by the environment.
        combined_stats.update({
            "info_at_done/{}".format(key):
                np.mean(self.info_at_done[key][-100:])
            for key in self.info_at_done.keys()
        })

        # Save combined_stats in a csv file.
        file_path = os.path.join(self.log_dir, "train.csv")
        exists = os.path.exists(file_path)
        with open(file_path, "a") as f:
            w = csv.DictWriter(f, fieldnames=combined_stats.keys())
            if not exists:
                w.writeheader()
            w.writerow(combined_stats)

        # Print statistics.
        print("-" * 67)
        for key in sorted(combined_stats.keys()):
            val = combined_stats[key]
            print("| {:<30} | {:<30} |".format(key, val))
        print("-" * 67)
        print("")


def main(args):
    """Run the imitation operation on a number of seeds."""
    # Parse input arguments.
    flags = parse_args(args)

    for i in range(flags.n_training):
        # Setup the seed value.
        seed = flags.seed + i
        print("Setting random seed to: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # The time when the current experiment started.
        now = strftime("%Y-%m-%d-%H:%M:%S")

        # Create a save directory folder (if it doesn't exist).
        dir_name = os.path.join("imitation_data", "{}/{}".format(
            flags.env_name, now))
        ensure_dir(dir_name)

        if flags.alg_cls == "DAgger":
            alg_params = dict(
                learning_rate=flags.learning_rate,
                l2_penalty=flags.l2_penalty,
                buffer_size=flags.buffer_size,
                num_train_steps=flags.num_train_steps,
            )
        else:
            alg_params = {
                "lambda": getattr(flags, "lambda"),
                "gae_gamma": flags.gae_gamma,
                "gae_lambda": flags.gae_lambda,
                "epsilon": flags.epsilon,
                "max_kl": flags.max_kl,
                "cg_damping": flags.cg_damping,
                "normalize_advantage": flags.normalize_advantage,
            }

        # Get the hyperparameters.
        hp = dict(
            env_name=flags.env_name,
            alg_cls=flags.alg_cls,
            alg_params=alg_params,
            model_params=dict(
                layers=getattr(flags, "model_params:layers"),
                dropout=getattr(flags, "model_params:dropout"),
                stochastic=getattr(flags, "model_params:stochastic"),
            ),
            num_rollouts=flags.num_rollouts,
            num_iterations=flags.num_iterations,
        )

        # Add the date/time for logging purposes.
        params_with_extra = hp.copy()
        params_with_extra['date/time'] = now
        params_with_extra['seed'] = seed

        # Add the hyperparameters to the folder.
        with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
            json.dump(params_with_extra, f, sort_keys=True, indent=4)

        # Create the trainer object.
        trainer = Trainer(log_dir=dir_name, **hp)

        # Start the training operation.
        trainer.learn()


if __name__ == '__main__':
    main(sys.argv[1:])
