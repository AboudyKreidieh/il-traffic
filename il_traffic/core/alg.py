"""Implementation of the DAgger algorithm.

See: https://arxiv.org/pdf/1011.0686.pdf
"""
import tensorflow as tf
import numpy as np
import random
import ray
import os
import csv
import time
from collections import defaultdict
from collections import deque
from tqdm import tqdm

from flow.core.util import ensure_dir

from il_traffic.utils.misc import dict_update
from il_traffic.utils.tf_util import make_session


# =========================================================================== #
#                    Model parameters for FeedForwardModel                    #
# =========================================================================== #

FEEDFORWARD_PARAMS = dict(
    # the size of the neural network for the policy
    layers=[32, 32, 32],
    # the activation function to use in the neural network
    act_fun=tf.nn.relu,
    # the model learning rate
    learning_rate=1e-3,
    # whether to enable batch normalization
    batch_norm=False,
    # whether to enable dropout
    dropout=False,
    # scale for the L2 regularization penalty
    l2_penalty=0,
    # whether the model should be stochastic or deterministic
    stochastic=False,
    # the number of ensemble models to use
    num_ensembles=1,
)


class DAgger(object):
    """DAgger training algorithm.

    See: https://arxiv.org/pdf/1011.0686.pdf

    Attributes
    ----------
    env_name : str
        the name of the training environment
    model_cls : type [ il_traffic.FeedForwardModel ]
        the type of model used during training
    num_envs : int
        number of environments used to run simulations in parallel. Each
        environment is run on a separate CPUS and uses the same policy as the
        rest. Must be less than or equal to num_rollouts.
    render : bool
        whether to render the training environment
    batch_size : int
        the number of elements in a batch when performing SGD
    buffer_size : int
        the maximum number of samples to store
    prob_add : float
        the probability of adding any given sample to the buffer of training
        samples
    num_rollouts : int
        number of rollouts to collect in between training iterations for the
        data aggregation procedure.
    num_train_steps : int
        number of times a training operation is run in a given iteration of
        training
    num_iterations : int
        number of training iterations
    seed : int
        the random seed for numpy, tensorflow, and random
    log_dir : str
        the directory where the training statistics and tensorboard log should
        be stored
    saver : tf.compat.v1.train.Saver
        tensorflow saver object
    info_at_done : dict < list >
        the results from the info dict when done masked were achieved in any
        given rollout. Used for logging purposes.
    total_steps : int
        the current number of samples collected
    sampler : Sampler or RaySampler
        the training environment sampler object. Each environment is provided
        a separate CPU to run on.
    obs : list of array_like
        the most recent training observation. One element for each environment.
    samples : dict
        a dictionary of samples, consisting of the following terms:

        * obs (list of array_like) : list of observation collected from various
          rollouts
        * actions (list of array_like) : list of actions from the expert policy
        * goals (list of array_like) : list of goals from the expert policy
    returns : list of float
        a list of cumulative returns from all rollouts. Used for logging
        purposes.
    ac_space : gym.spaces.Box
        the action space of the environment
    ob_space : gym.spaces.Box
        the observation space of the environment
    model_params : dict
        dictionary of model-specific parameters
    graph : tf.compat.v1.Graph
        the tensorflow graph
    model : il_traffic.FeedForwardModel
        the model to imitate on
    sess : tf.compat.v1.Session
        the tensorflow session
    info_ph : dict of tf.compat.v1.placeholder
        placeholder for the info_dict at done. Used for logging purposes.
    rew_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last epoch. Used
        for logging purposes.
    rew_history_ph : tf.compat.v1.placeholder
        a placeholder for the average training return for the last 100
        episodes. Used for logging purposes.
    trainable_vars : list of str
        the names of the trainable variables. Used when creating the saver
        object.
    """

    def __init__(self,
                 env_name,
                 model_cls,
                 expert,
                 num_envs,
                 render,
                 batch_size,
                 buffer_size,
                 prob_add,
                 num_rollouts,
                 num_train_steps,
                 num_iterations,
                 initial_episodes,
                 seed,
                 log_dir,
                 env_params,
                 model_params):
        """Instantiate the DAgger object.

        Parameters
        ----------
        env_name : str
            the name of the training environment
        model_cls : type [ il_traffic.FeedForwardModel ]
            the type of model used during training
        expert : int
            the expert policy used. Must be one of:
             0) Intelligent Driver Model
             1) FollowerStopper
             2) PISaturation
             3) TimeHeadwayFollowerStopper
        num_envs : int
            number of environments used to run simulations in parallel. Each
            environment is run on a separate CPUS and uses the same policy as
            the rest. Must be less than or equal to num_rollouts.
        render : bool
            whether to render the training environment
        batch_size : int
            the number of elements in a batch when performing SGD
        buffer_size : int
            the maximum number of samples to store
        prob_add : float
            the probability of adding any given sample to the buffer of
            training samples
        num_rollouts : int
            number of rollouts to collect in between training iterations for
            the data aggregation procedure.
        num_train_steps : int
            number of times a training operation is run in a given iteration
            of training
        num_iterations : int
            number of training iterations
        initial_episodes : int
            initial number of episodes to collect from the expert policy
        env_params : dict
            dictionary of environment-specific parameters. May contain the
            following variables:

            * obs_frames (int): number of observation frames to use. Additional
              frames are provided from previous time steps.
            * frame_skip (int): frames to ignore in between each delta
              observation
            * full_history (bool): whether to use all observations from
              previous steps. If set to False, only the past speed is used.
            * avg_speed (bool): whether to include the average speed of the
              leader vehicle in the observation
        model_params : dict or None
            dictionary of model-specific parameters. If set to None, default
            parameters are provided.
        seed : int
            the random seed for numpy, tensorflow, and random
        log_dir : str
            the directory where the training statistics and tensorboard log
            should be stored
        """
        self.env_name = env_name
        self.model_cls = model_cls
        self.num_envs = num_envs
        self.render = render
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prob_add = prob_add
        self.num_rollouts = num_rollouts
        self.num_train_steps = num_train_steps
        self.num_iterations = num_iterations
        self.env_params = env_params
        self.seed = seed
        self.log_dir = log_dir

        # a few initializations
        self.saver = None
        self.info_at_done = defaultdict(list)
        self.total_steps = 0

        # Instantiate the ray instance.
        if num_envs > 1:
            ray.init(num_cpus=num_envs+1, ignore_reinit_error=True)

        # Create the tensorflow graph and session.
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = make_session(num_cpu=3, graph=self.graph)

        # Create the environment and collect the initial observations.
        self.sampler, self.obs = self._setup_sampler(
            env_name=env_name,
            render=render,
            expert=expert,
            env_params=env_params,
        )

        # Generate initial expert observations and actions.
        self.samples = {
            "obs": deque(maxlen=self.buffer_size),
            "actions": deque(maxlen=self.buffer_size),
            "goals": deque(maxlen=self.buffer_size),
        }
        self.returns = []
        self._info_keys = None
        self._load_expert_data(initial_episodes)

        # Collect the spaces of the environments.
        self.ac_space, self.ob_space = self._get_spaces()

        # Prepare the model params.
        self.model_params = {}
        self.model_params.update(FEEDFORWARD_PARAMS)
        self.model_params = dict_update(self.model_params, model_params or {})

        # Create the model variables and operations.
        self.model = None
        self.info_ph = {}
        self.rew_ph = None
        self.rew_history_ph = None
        self.trainable_vars = self._setup_model()

    def learn(self):
        """Perform the complete training operation."""
        start_time = time.time()

        # Create a saver object.
        self.saver = tf.compat.v1.train.Saver(
            self.trainable_vars,
            max_to_keep=self.num_iterations)

        # Make sure that the log directory exists, and if not, make it.
        ensure_dir(self.log_dir)
        ensure_dir(os.path.join(self.log_dir, "checkpoints"))

        # Create a tensorboard object for logging.
        save_path = os.path.join(self.log_dir, "tb_log")
        writer = tf.compat.v1.summary.FileWriter(save_path)

        # file path for training and evaluation results
        train_filepath = os.path.join(self.log_dir, "train.csv")

        # Setup the seed value.
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

        with self.sess.as_default(), self.graph.as_default():
            for itr in range(self.num_iterations):
                print("Iteration {}:\n".format(itr))

                # Optimization steps.
                print("- Training model.")
                loss = self._train()

                # Simulation steps.
                print("\n- Generating samples from model.")
                self._generate_rollouts(use_expert=False)

                # Log and print results to csv.
                self._log_results(train_filepath, start_time, itr, loss)

                # Update tensorboard summary.
                self._log_tb(writer)

                # Save a checkpoint of the model.
                self.save(os.path.join(self.log_dir, "checkpoints/itr"))

    def save(self, save_path):
        """Save the parameters of a tensorflow model.

        Parameters
        ----------
        save_path : str
            Prefix of filenames created for the checkpoint
        """
        self.saver.save(self.sess, save_path, global_step=self.total_steps)

    def load(self, load_path):
        """Load model parameters from a checkpoint.

        Parameters
        ----------
        load_path : str
            location of the checkpoint
        """
        self.saver.restore(self.sess, load_path)

    def _setup_sampler(self, env_name, render, expert, env_params):
        """Create the environment and collect the initial observations.

        Parameters
        ----------
        env_name : str
            the name of the environment
        render : bool
            whether to render the environment
        expert : int
            the expert policy used
        env_params : dict
            dictionary of environment-specific parameters

        Returns
        -------
        list of Sampler or list of RaySampler
            the sampler objects
        list of array_like or list of dict < str, array_like >
            the initial observation. If the environment is multi-agent, this
            will be a dictionary of observations for each agent, indexed by the
            agent ID. One element for each environment.
        """
        if self.num_envs > 1:
            from il_traffic.utils.sampler import RaySampler
            sampler = [RaySampler.remote(
                env_name=env_name,
                render=render,
                expert=expert,
                env_params=env_params,
                env_num=env_num)
                for env_num in range(self.num_envs)]
            obs = ray.get([s.get_init_obs.remote() for s in sampler])
        else:
            from il_traffic.utils.sampler import Sampler
            sampler = [Sampler(
                env_name=env_name,
                render=render,
                expert=expert,
                env_params=env_params,
                env_num=0)]
            obs = [s.get_init_obs() for s in sampler]

        return sampler, obs

    def _get_spaces(self):
        """Collect the spaces of the environments.

        Returns
        -------
        gym.spaces.*
            the action space of the training environment
        gym.spaces.*
            the observation space of the training environment
        """
        sampler = self.sampler[0]

        if self.num_envs > 1:
            ac_space = ray.get(sampler.action_space.remote())
            ob_space = ray.get(sampler.observation_space.remote())
        else:
            ac_space = sampler.action_space()
            ob_space = sampler.observation_space()

        return ac_space, ob_space

    def _load_expert_data(self, num_episodes):
        """Generate expert data."""
        print("Generating initial expert data.")
        self._generate_rollouts(use_expert=True, num_episodes=num_episodes)

        # Reset number of training steps.
        self.total_steps = 0

    def _setup_model(self):
        """Create the graph, session, model, and summary objects."""
        with self.graph.as_default():
            # Create the model.
            self.model = self.model_cls(
                sess=self.sess,
                ob_dim=self.ob_space.shape[0],
                ac_dim=self.ac_space.shape[0],
                **self.model_params,
            )

            # for tensorboard logging
            with tf.compat.v1.variable_scope("Train"):
                self.rew_ph = tf.compat.v1.placeholder(tf.float32)
                self.rew_history_ph = tf.compat.v1.placeholder(tf.float32)

            # Add tensorboard scalars for the return, return history, and
            # success rate.
            tf.compat.v1.summary.scalar(
                "Train/return", self.rew_ph)
            tf.compat.v1.summary.scalar(
                "Train/return_history", self.rew_history_ph)

            # Add the info_dict various to tensorboard as well.
            with tf.compat.v1.variable_scope("info_at_done"):
                for key in self._info_keys:
                    self.info_ph[key] = tf.compat.v1.placeholder(
                        tf.float32, name="{}".format(key))
                    tf.compat.v1.summary.scalar(
                        "{}".format(key), self.info_ph[key])

            # Create the tensorboard summary.
            self.summary = tf.compat.v1.summary.merge_all()

            # Initialize the model parameters and optimizers.
            with self.sess.as_default():
                self.sess.run(tf.compat.v1.global_variables_initializer())

            return tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    def _generate_rollouts(self, use_expert, num_episodes=None):
        """Create a number of rollouts and collect expert data.

        Parameters
        ----------
        use_expert : bool
            whether to use the expert for performing the action within the
            environment
        num_episodes : int or None
            number of episodes to generate. Defaults to self.num_rollouts
        """
        num_rollouts = 0
        done = [False for _ in range(self.num_envs)]
        totalr = [0. for _ in range(self.num_envs)]

        while num_rollouts < (num_episodes or self.num_rollouts):
            # Compute the action to be performed. If using the expert, the
            # function is embedded within the environment.
            if use_expert:
                action = [None for _ in range(self.num_envs)]
            else:
                action = [
                    self.model.get_action(self.obs[i], env_num=i)
                    if not done[i] else None for i in range(self.num_envs)]

            # Advance the simulation by one step.
            if self.num_envs > 1:
                ret = ray.get([
                    self.sampler[env_num].collect_sample.remote(
                        obs=self.obs[env_num], action=action[env_num])
                    for env_num in range(self.num_envs)
                    if use_expert  # collecting expert data
                    or action[env_num] is not None  # env not terminated
                ])
            else:
                ret = [self.sampler[0].collect_sample(
                    obs=self.obs[0], action=action[0])]

            self.total_steps += len(ret)

            for ret_i in ret:
                num = ret_i["env_num"]
                expert_action = ret_i["expert_action"]
                expert_goal = ret_i["expert_goal"]
                reward = ret_i["reward"]
                obs0 = ret_i["obs0"]
                obs1 = ret_i["obs1"]
                done[num] = ret_i["done"]
                info = ret_i["info"]

                # Update current state data and internal storages.
                totalr[num] += reward
                self.obs[num] = obs1

                # Store a subset of information in the buffer.
                for i in range(len(obs0)):
                    # Don't add the next sample with some probability.
                    if random.uniform(0, 1) > self.prob_add:
                        continue
                    # store observation
                    self.samples["obs"].append(obs0[i])
                    # store expert action
                    self.samples["actions"].append(expert_action[i])
                    # store expert goal
                    self.samples["goals"].append(expert_goal[i])

                # Handle episode done.
                if done[num]:
                    # Reset the model.
                    if not use_expert:
                        self.model.reset(num)
                        self.returns.append(totalr[num])
                    totalr[num] = 0
                    num_rollouts += 1

                    # Fill in the info keys term if it has not been set yet.
                    if self._info_keys is None:
                        self._info_keys = list(info.keys())

                    # Store the info value at the end of the rollout.
                    for key in info.keys():
                        self.info_at_done[key].append(info[key])

                    # Stop collecting samples if you just need to wait for
                    # other rollouts to finish.
                    if done and num_rollouts + sum([not d for d in done]) \
                            < self.num_rollouts:
                        done[num] = False

    def _train(self):
        """Perform a number of optimization steps.

        Returns
        -------
        list of array_like
            the losses associated with the actions from each training step, and
            each ensemble
        """
        # Covert samples to numpy array.
        observations = np.array(self.samples["obs"])
        actions = np.array(self.samples["actions"])

        loss = []
        for _ in tqdm(range(self.num_train_steps)):
            # Sample a batch.
            batch_i = [np.random.randint(0, len(actions), size=self.batch_size)
                       for _ in range(self.model.num_ensembles)]

            # Run the training step.
            loss_step = self.model.train(
                obs=[observations[batch_i[i], :]
                     for i in range(self.model.num_ensembles)],
                action=[actions[batch_i[i], :]
                        for i in range(self.model.num_ensembles)],
            )
            loss.append(loss_step)

        return loss

    def _log_results(self, file_path, start_time, train_itr, loss):
        """Log training statistics.

        Parameters
        ----------
        file_path : str
            the list of cumulative rewards from every episode in the evaluation
            phase
        start_time : float
            the time when training began. This is used to print the total
            training time.
        train_itr : int
            the training iteration
        loss : list of list of float
            imitation loss from most recent epoch
        """
        # Log statistics.
        duration = time.time() - start_time

        combined_stats = {
            "rollout/return": np.mean(self.returns[-self.num_rollouts:]),
            "rollout/return_history": np.mean(self.returns[-100:]),
            "total/epochs": train_itr + 1,
            "total/episodes": (train_itr + 1) * self.num_rollouts,
            "total/steps": self.total_steps,
            "total/duration": duration,
            "total/steps_per_second": self.total_steps / duration,
        }

        # Add the imitation loss for each model.
        loss = np.asarray(loss)
        combined_stats.update({
            "rollout/imitation_loss_{}".format(i): np.mean(loss[:, i])
            for i in range(np.shape(loss)[1])})

        # Information passed by the environment.
        combined_stats.update({
            "info_at_done/{}".format(key):
                np.mean(self.info_at_done[key][-100:])
            for key in self.info_at_done.keys()
        })

        # Save combined_stats in a csv file.
        if file_path is not None:
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

    def _log_tb(self, writer):
        """Run and store summary."""
        # Get feed_dict for model attributes.
        td_map = self.model.get_td_map(
            obs=np.array(self.samples["obs"]),
            action=np.array(self.samples["actions"]),
        )

        # average epoch and running cumulative returns
        td_map.update({
            self.rew_ph: np.mean(self.returns[-self.num_rollouts:]),
            self.rew_history_ph: np.mean(self.returns[-100:]),
        })

        # average epoch and running info_dict values
        td_map.update({
            self.info_ph[key]: np.mean(self.info_at_done[key])
            for key in self.info_ph.keys()
        })

        # Run tensorflow operation.
        summary = self.sess.run(self.summary, td_map)
        writer.add_summary(summary, self.total_steps)