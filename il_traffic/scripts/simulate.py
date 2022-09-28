"""Run a simulation.

Usage
    python simulate.py --network_type "i24" --n_vehicles 100
"""
import sys
import argparse
import warnings
import numpy as np
from time import strftime

from il_traffic.environments.trajectory import TrajectoryEnv
from il_traffic.environments.bottleneck import BottleneckEnv

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

    # optional input parameters
    parser.add_argument(
        '--network_type',
        type=str,
        default='i24',
        help='the type of network to simulate. One of {"bottleneck", "i24"}.')
    parser.add_argument(
        '--n_vehicles',
        type=int,
        default=100,
        help='total number of vehicles')
    parser.add_argument(
        '--av_penetration',
        type=float,
        default=0,
        help='AV penetration rate')
    parser.add_argument(  # TODO
        '--save_video',
        action='store_true',
        help='whether to save the frames of the GUI. These can be processed '
             'and coupled together later to generate a video of the '
             'simulation.')

    return parser.parse_args(args)


def main(args):
    """Run the simulation of the expert policy."""
    # Parse command-line arguments.
    flags = parse_args(args)

    # The time when the current experiment started.
    now = strftime("%Y-%m-%d-%H:%M:%S")

    # Create a save directory folder (if it doesn't exist).
    dir_name = 'simulate/{}/{}'.format(flags.network_type, now)

    # Create the environment.
    if flags.network_type == "i24":
        env = TrajectoryEnv(
            n_vehicles=flags.n_vehicles,
            av_penetration=flags.av_penetration,
        )
    elif flags.network_type == "bottleneck":
        env = BottleneckEnv(
            n_vehicles=flags.n_vehicles,
            av_penetration=flags.av_penetration,
        )
    else:
        raise NotImplementedError("Unknown network type: {}".format(
            flags.network_type))

    # Reset environment.
    env.reset()

    # Run the simulation.
    done = False
    info = {}
    while not done:
        _, _, done, info = env.step(action=None)

    # Compute/display statistics.
    for key in info.keys():
        if key != "expert_action":
            print(key, info[key])

    # Generate emission files.
    env.gen_emission(flags.network_type, dir_name)


if __name__ == "__main__":
    main(sys.argv[1:])
