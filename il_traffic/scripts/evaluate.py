"""Evaluate the performance of a trained model."""
import argparse
import os
import json
import sys
import torch

from il_traffic.models.fcnet import FeedForwardModel
from il_traffic.scripts.simulate import HIGHWAY_PARAMS
from il_traffic.scripts.simulate import rollout
from il_traffic.scripts.simulate import plot_results
from il_traffic.utils.flow_utils import get_base_env_params
from il_traffic.utils.flow_utils import create_env
from il_traffic.utils.misc import ensure_dir


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a Flow simulation of "
                    "an expert policy.",
        epilog="python simulate.py")

    # required arguments
    parser.add_argument(
        'model_path',
        type=str,
        help='the path to the save model information')

    # optional input parameters
    parser.add_argument(
        '--ckpt_num',
        type=int,
        default=None,
        help='the checkpoint number. If not specified, the last checkpoint is '
             'used.')
    parser.add_argument(
        '--inflow',
        type=float,
        default=HIGHWAY_PARAMS["inflow"],
        help='the inflow rate of vehicles (human and automated)')
    parser.add_argument(
        '--end_speed',
        type=float,
        default=HIGHWAY_PARAMS["end_speed"],
        help='the maximum speed at the downstream boundary edge')
    parser.add_argument(
        '--penetration_rate',
        type=float,
        default=HIGHWAY_PARAMS["penetration_rate"],
        help='penetration rate of the AVs. 0.10 corresponds to 10%.')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Specifies whether to render the simulation during runtime.')
    parser.add_argument(
        '--use_warmup',
        action='store_true',
        help='specifies whether to use a warmup file when initializing a '
             'simulation.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='whether to save the frames of the GUI. These can be processed '
             'and coupled together later to generate a video of the '
             'simulation.')

    return parser.parse_known_args(args)[0]


def create_custom_expert(env, model_path, ckpt_num=None):
    """Return a functional form of the expert policy.

    Parameters
    ----------
    env : gym.Env
        the evaluation environment
    model_path : str
        the path to the save model information
    ckpt_num : int or None
        the checkpoint number. If set to None, the most recent checkpoint is
        used.

    Returns
    -------
    function
        the functional form of the policy
    """
    # Create the torch device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect the model parameters.
    with open(os.path.join(model_path, "hyperparameters.json"), "r") as f:
        hyperparameters = json.load(f)
        model_params = hyperparameters["model_params"]

    # Create the model.
    model = FeedForwardModel(
        ob_dim=env.observation_space.shape[0],
        ac_dim=env.action_space.shape[0],
        **model_params,
    )

    # Choose the last checkpoint if a value was not specified. TODO
    if ckpt_num is None:
        filenames = os.listdir(os.path.join(model_path, "checkpoints"))
        metafiles = [f[:-5] for f in filenames if f[-5:] == ".meta"]
        metanum = [int(f.split("-")[-1]) for f in metafiles]
        ckpt_num = max(metanum)

    # Load the learned model parameters. TODO
    saver.restore(
        sess,
        os.path.join(model_path, "checkpoints/itr-{}".format(ckpt_num)),
    )

    return lambda x: model(x)


def main(args):
    """Run the simulation of the expert policy."""
    # Parse command-line arguments.
    flags = parse_args(args)

    # Collect the hyperparameters from the stored data.
    with open(os.path.join(flags.model_path, "hyperparameters.json"), "r") \
            as f:
        hyperparameters = json.load(f)
    env_params = hyperparameters["env_params"]
    network_type = hyperparameters["env_name"]

    # Get the network parameters.
    network_params = dict(
        inflow=flags.inflow,
        end_speed=flags.end_speed,
        penetration_rate=flags.penetration_rate,
    )

    # Specify the parameters necessary to properly control the automated
    # vehicles.
    environment_params = get_base_env_params(
        network_type=network_type,
        controller_type=0,
        noise=0,
        save_video=flags.save_video,
    )

    # Update the environment parameters based on the overriding values from
    # the algorithm class.
    if "obs_frames" in env_params:
        environment_params["obs_frames"] = env_params["obs_frames"]
    if "frame_skip" in env_params:
        environment_params["frame_skip"] = env_params["frame_skip"]
    if "avg_speed" in env_params:
        environment_params["avg_speed"] = env_params["avg_speed"]
    if "v_des" in env_params:
        environment_params["v_des"] = env_params["v_des"]

    # Specify an emission path.
    inflow = network_params["inflow"]
    end_speed = network_params["end_speed"]
    emission_path = os.path.join(flags.model_path, "results/{}-{}".format(
        int(inflow), int(end_speed)))
    ensure_dir(emission_path)

    # Create the environment.
    env = create_env(
        network_type=network_type,
        network_params=network_params,
        environment_params=environment_params,
        render=flags.render,
        emission_path=emission_path if flags.gen_emission else None,
        use_warmup=flags.use_warmup,
    )

    # Create the expert model.
    model = create_custom_expert(env, flags.model_path, flags.ckpt_num)

    # Set the end speed if using the I-210 network.
    if network_type == "i210":
        env.k.kernel_api.edge.setMaxSpeed("119257908#3", flags.end_speed)

    # Execute a rollout.
    mpg_val, mpg_time = rollout(env=env, model=model, save_path=emission_path)

    # Plot the results from the simulation.
    if flags.gen_emission:
        plot_results(
            emission_path=emission_path,
            network_type=network_type,
            mpg_vals=mpg_val,
            mpg_times=mpg_time,
            t_total=(env.env_params.horizon +
                     env.env_params.warmup_steps) * env.sim_step,
            use_warmup=flags.use_warmup,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
