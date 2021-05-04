"""Script containing the FeedForwardModel object."""
import tensorflow as tf
import numpy as np

from il_traffic.utils.tf_util import reduce_std
from il_traffic.utils.tf_util import create_fcnet
from il_traffic.utils.tf_util import get_trainable_vars
from il_traffic.utils.tf_util import print_params_shape


class FeedForwardModel(object):
    """Fully-connected model object.

    Attributes
    ----------
    layers : list of int
        the size of the neural network for the policy
    learning_rate : float
        the model learning rate
    learning_rate : float
        the model learning rate
    act_fun : tf.nn.*
        the activation function to use in the neural network
    batch_norm : bool
        whether to enable batch normalization
    dropout : bool
        whether to enable dropout
    l2_penalty : float
        scale for the L2 regularization penalty
    stochastic : bool
        whether the policy is stochastic or deterministic
    num_ensembles : int
        the number of ensemble models to use
    obs_ph : list of tf.Placeholder. One for each ensemble.
        the input placeholder
    output : list of tf.Variable
        the output from the model. One for each ensemble.
    phase_ph : list of tf.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing. One
        for each ensemble.
    rate_ph : list of tf.placeholder
        the probability with which elements in a model are set to 0. One for
        each ensemble.
    expert_ph : list of tf.placeholder
        placeholder for the expert action. One for each ensemble.
    loss : list of tf.Variable
        the output from the loss of a model. One for each ensemble.
    train_step : list of tf.Operation
        the training operation for a model. One for each ensemble.
    """

    def __init__(self,
                 sess,
                 ob_dim,
                 ac_dim,
                 layers,
                 learning_rate,
                 act_fun,
                 batch_norm,
                 dropout,
                 l2_penalty,
                 stochastic,
                 num_ensembles,
                 base=None):
        """Instantiate the model.

        Parameters
        ----------
        sess : tf.compat.v1.Session
            the tensorflow session
        ob_dim : int
            the number of elements in the observation
        ac_dim : int
            the number of elements in the action
        layers : list of int
            the size of the neural network for the policy
        learning_rate : float
            the model learning rate
        act_fun : tf.nn.*
            the activation function to use in the neural network
        batch_norm : bool
            whether to enable batch normalization
        dropout : bool
            whether to enable dropout
        l2_penalty : float
            scale for the L2 regularization penalty
        stochastic : bool
            whether the policy is stochastic or deterministic
        num_ensembles : int
            the number of ensemble models to use
        """
        self.sess = sess
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.layers = layers
        self.learning_rate = learning_rate
        self.act_fun = act_fun
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.l2_penalty = l2_penalty
        self.stochastic = stochastic
        self.num_ensembles = num_ensembles

        # Log variance bounds (for stochastic models).
        self._max_logvar = tf.Variable(
            -3 * np.ones([1, self.ac_dim]), dtype=tf.float32)
        self._min_logvar = tf.Variable(
            -7 * np.ones([1, self.ac_dim]), dtype=tf.float32)

        # One for each ensemble.
        self.output = []
        self.obs_ph = []
        self.phase_ph = []
        self.rate_ph = []
        self.expert_ph = []
        self.loss = []
        self.train_step = []

        print("Creating model:")
        for i in range(num_ensembles):
            scope_i = "" if base is None else "{}/".format(base)
            scope_i += "{}".format(i)
            with tf.compat.v1.variable_scope(scope_i, reuse=False):
                # Create the input placeholders.
                self.obs_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    name='obs_ph',
                    shape=[None, ob_dim]))
                self.phase_ph.append(tf.compat.v1.placeholder(
                    tf.bool,
                    name='phase'))
                self.rate_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    name='rate'))

                # Create the model.
                output = create_fcnet(
                    obs=self.obs_ph[-1],
                    layers=layers,
                    num_output=ac_dim,
                    stochastic=stochastic,
                    act_fun=act_fun,
                    batch_norm=batch_norm,
                    phase=self.phase_ph[-1],
                    dropout=dropout,
                    rate=self.rate_ph[-1],
                    scope="Model",
                    reuse=False,
                )

                # Print the shape and parameters of the model.
                print_params_shape(
                    "{}/Model/".format(scope_i), "Model {}".format(i))

                if stochastic:
                    # Get the mean and log-var.
                    mean, log_var = output

                    # Bind the variance by desired values.
                    log_var = self._max_logvar - tf.nn.softplus(
                        self._max_logvar - log_var)
                    log_var = self._min_logvar + tf.nn.softplus(
                        log_var - self._min_logvar)

                    # Sample and action from the above variables.
                    std = tf.sqrt(tf.exp(log_var))
                    self.output.append(mean + std * tf.random_normal(
                        shape=tf.shape(mean),
                        dtype=tf.float32,
                    ))
                else:
                    mean = None
                    log_var = None
                    self.output.append(output)

                # Create the target placeholder.
                self.expert_ph.append(tf.compat.v1.placeholder(
                    tf.float32,
                    name='expert_ph',
                    shape=[None, self.ac_dim]))

                # Create the loss function.
                if stochastic:
                    var = tf.exp(log_var)
                    inv_var = tf.divide(1, var)
                    norm_output = mean - self.expert_ph[-1]

                    loss = tf.multiply(tf.multiply(norm_output, inv_var),
                                       norm_output)
                    loss = tf.reduce_sum(loss, axis=1)
                    loss += tf.math.log(tf.math.reduce_prod(var, axis=1))
                    self.loss.append(loss)
                else:
                    self.loss.append(tf.reduce_mean(tf.square(
                        self.expert_ph[-1] - self.output[-1])))

                # Add a regularization penalty.
                self.loss[-1] += self._l2_loss(
                    self.l2_penalty, "{}/Model".format(scope_i))

                # Create the training operation.
                self.train_step.append(tf.compat.v1.train.AdamOptimizer(
                    self.learning_rate).minimize(
                        self.loss[-1],
                        var_list=get_trainable_vars("{}/Model".format(
                            scope_i)),
                    )
                )

        # Setup the tensorboard statistics for the model.
        self._setup_stats(base or "Model")

    def get_action(self, obs, env_num):
        """Compute the action by the agent.

        Parameters
        ----------
        obs : array_like
            the input observation
        env_num : int
            the environment number. Used by hierarchical policies

        Returns
        -------
        array_like
            the output action
        """
        feed_dict = {}
        for i in range(self.num_ensembles):
            feed_dict.update({
                self.obs_ph[i]: obs,
                self.phase_ph[i]: 0,
                self.rate_ph[i]: 0.0,
            })

        actions = self.sess.run(self.output, feed_dict=feed_dict)

        return np.asarray(actions).mean(axis=0)

    def train(self, obs, action):
        """Perform the training operation.

        Parameters
        ----------
        obs : array_like
            a batch of observations
        action : array_like
            a batch of expert actions

        Returns
        -------
        float
            the estimated imitation loss
        """
        feed_dict = {}
        for i in range(self.num_ensembles):
            feed_dict.update({
                self.expert_ph[i]: action[i],
                self.obs_ph[i]: obs[i],
                self.phase_ph[i]: 1,
                self.rate_ph[i]: 0.5,
            })

        v = self.sess.run(self.loss + self.train_step, feed_dict=feed_dict)

        return [v[i] for i in range(self.num_ensembles)]

    def get_td_map(self, obs, action):
        """Return dict map for the summary (to be run in the algorithm).

        Parameters
        ----------
        obs : array_like
            a batch of observations
        action : array_like
            a batch of expert actions

        Returns
        -------
        dict
            the td map
        """
        td_map = {}
        for i in range(self.num_ensembles):
            td_map.update({
                self.expert_ph[i]: action,
                self.obs_ph[i]: obs,
                self.phase_ph[i]: 0,
                self.rate_ph[i]: 0.0,
            })

        return td_map

    def _setup_stats(self, base):
        """Create the running means and std of the model inputs and outputs.

        This method also adds the same running means and stds as scalars to
        tensorboard for additional storage.
        """
        ops = []
        names = []

        for i in range(self.num_ensembles):
            ops += [tf.reduce_mean(self.output[i])]
            names += ['{}/model_action_mean_{}'.format(base, i)]
            ops += [reduce_std(self.output[i])]
            names += ['{}/model_action_std_{}'.format(base, i)]

        ops += [tf.reduce_mean(self.expert_ph[0])]
        names += ['{}/expert_action_mean'.format(base)]
        ops += [reduce_std(self.expert_ph[0])]
        names += ['{}/expert_action_std'.format(base)]

        # Add all names and ops to the tensorboard summary.
        for op, name in zip(ops, names):
            tf.compat.v1.summary.scalar(name, op)

    @staticmethod
    def _l2_loss(l2_penalty, scope_name):
        """Compute the L2 regularization penalty.

        Parameters
        ----------
        l2_penalty : float
            L2 regularization penalty
        scope_name : str
            the scope of the trainable variables to regularize

        Returns
        -------
        float
            the overall regularization penalty
        """
        if l2_penalty > 0:
            print("regularizing policy network: L2 = {}".format(l2_penalty))
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=l2_penalty, scope="{}/l2_regularize".format(scope_name))
            l2_loss = tf.contrib.layers.apply_regularization(
                regularizer,
                weights_list=get_trainable_vars(scope_name))
        else:
            # no regularization
            l2_loss = 0

        return l2_loss
