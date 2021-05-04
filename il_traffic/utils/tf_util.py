"""TensorFlow utility methods."""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import reduce


def make_session(num_cpu, graph=None):
    """Return a session that will use <num_cpu> CPU's only.

    Parameters
    ----------
    num_cpu : int
        number of CPUs to use for TensorFlow
    graph : tf.Graph
        the graph of the session

    Returns
    -------
    tf.compat.v1.Session
        a tensorflow session
    """
    tf_config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)

    # Prevent tensorflow from taking all the gpu memory.
    tf_config.gpu_options.allow_growth = True

    return tf.compat.v1.Session(config=tf_config, graph=graph)


def get_trainable_vars(name=None):
    """Return the trainable variables.

    Parameters
    ----------
    name : str
        the scope

    Returns
    -------
    list of tf.Variable
        trainable variables
    """
    return tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


def print_params_shape(scope, param_type):
    """Print parameter shapes and number of parameters.

    Parameters
    ----------
    scope : str
        scope containing the parameters
    param_type : str
        the name of the parameter
    """
    shapes = [var.get_shape().as_list() for var in get_trainable_vars(scope)]
    nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in shapes])
    print('  {} shapes: {}'.format(param_type, shapes))
    print('  {} params: {}'.format(param_type, nb_params))


def reduce_std(tensor, axis=None, keepdims=False):
    """Get the standard deviation of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor or tf.Variable
        the input tensor
    axis : int or list of int
        the axis to itterate the std over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the std of the tensor
    """
    return tf.sqrt(reduce_var(tensor, axis=axis, keepdims=keepdims))


def reduce_var(tensor, axis=None, keepdims=False):
    """Get the variance of a Tensor.

    Parameters
    ----------
    tensor : tf.Tensor
        the input tensor
    axis : int or list of int
        the axis to itterate the variance over
    keepdims : bool
        keep the other dimensions the same

    Returns
    -------
    tf.Tensor
        the variance of the tensor
    """
    tensor_mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    devs_squared = tf.square(tensor - tensor_mean)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def layer(val,
          num_outputs,
          name,
          act_fun=None,
          kernel_initializer=slim.variance_scaling_initializer(
              factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
          batch_norm=False,
          phase=None,
          dropout=False,
          rate=None):
    """Create a fully-connected layer.

    Parameters
    ----------
    val : tf.Variable
        the input to the layer
    num_outputs : int
        number of outputs from the layer
    name : str
        the scope of the layer
    act_fun : tf.nn.* or None
        the activation function
    kernel_initializer : Any
        the initializing operation to the weights of the layer
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability with which elements in a model are set to 0. Set to 0.5
        during training and 0.0 during testing

    Returns
    -------
    tf.Variable
        the output from the layer
    """
    val = tf.layers.dense(
        val, num_outputs, name=name, kernel_initializer=kernel_initializer)

    if batch_norm:
        val = tf.contrib.layers.batch_norm(
            val,
            center=True,
            scale=True,
            is_training=phase,
            scope='bn_{}'.format(name),
        )

    if act_fun is not None:
        val = act_fun(val)

    if dropout:
        val = tf.nn.dropout(val, rate=rate)

    return val


def create_fcnet(obs,
                 layers,
                 num_output,
                 stochastic,
                 act_fun,
                 batch_norm,
                 phase,
                 dropout,
                 rate,
                 scope=None,
                 reuse=False,
                 output_pre=""):
    """Create a fully-connected neural network model.

    Parameters
    ----------
    obs : tf.Variable
        the input to the model
    layers : list of int
        the size of the neural network for the model
    num_output : int
        number of outputs from the model
    stochastic : bool
        whether the model should be stochastic or deterministic
    act_fun : tf.nn.* or None
        the activation function
    batch_norm : bool
        whether to enable batch normalization
    phase : tf.compat.v1.placeholder
        a placeholder that defines whether training is occurring for the batch
        normalization layer. Set to True in training and False in testing.
    dropout : bool
        whether to enable dropout
    rate : tf.compat.v1.placeholder
        the probability with which elements in a model are set to 0
    scope : str
        the scope name of the model
    reuse : bool
        whether or not to reuse parameters
    output_pre : str
        an addition to the name of the output variable

    Returns
    -------
    tf.Variable or (tf.Variable, tf.Variable)
        the output from the model. a variable representing the output from the
        model in the deterministic case and a tuple of the (mean, logstd) in
        the stochastic case
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        pi_h = obs

        # Create the hidden layers.
        for i, layer_size in enumerate(layers):
            pi_h = layer(
                pi_h, layer_size, 'fc{}'.format(i),
                act_fun=act_fun,
                batch_norm=batch_norm,
                dropout=dropout,
                phase=phase,
                rate=rate,
            )

        if stochastic:
            # Create the output mean.
            policy_mean = layer(
                pi_h, num_output, 'mean',
                act_fun=None,
            )

            # Create the output log_std.
            log_std = layer(
                pi_h, num_output, 'log_std',
                act_fun=None,
            )

            policy = (policy_mean, log_std)
        else:
            # Create the output layer.
            policy = layer(
                pi_h, num_output, '{}output'.format(output_pre),
                act_fun=None,
            )

        return policy
