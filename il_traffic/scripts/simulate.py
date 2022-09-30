"""Run a simulation.

Usage
    python simulate.py --network_type "i24" --n_vehicles 100
"""
import json
import os
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
    parser.add_argument(
        '--bn_coeff',
        type=float,
        default=0.8,
        help='TODO')
    parser.add_argument(
        '--v_init',
        type=float,
        default=28.,
        help='TODO')
    parser.add_argument(
        '--c1',
        type=float,
        default=0.8,
        help='TODO')
    parser.add_argument(
        '--c2',
        type=float,
        default=28.,
        help='TODO')
    parser.add_argument(
        '--th_target',
        type=float,
        default=0.8,
        help='TODO')
    parser.add_argument(
        '--sigma',
        type=float,
        default=28.,
        help='TODO')
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

    if flags.network_type == "i24":
        # Create a save directory folder (if it doesn't exist).
        dir_name = 'simulate/{}/{}'.format(flags.network_type, now)

        # Create the environment.
        env = TrajectoryEnv(
            n_vehicles=flags.n_vehicles,
            av_penetration=flags.av_penetration,
        )
    elif flags.network_type == "bottleneck":
        # Create a save directory folder (if it doesn't exist).
        dir_name = \
            'simulate/{}/bn_coeff={}/pen={}'.format(
                flags.network_type,
                round(flags.bn_coeff, 1),
                # round(flags.v_init),
                flags.av_penetration,
            )

        if flags.av_penetration > 0:
            dir_name = os.path.join(
                dir_name,
                f"c1={flags.c1},c2={flags.c2},w={flags.sigma}", now)
        else:
            dir_name = os.path.join(dir_name, now)

        # Create the environment.
        env = BottleneckEnv(
            n_vehicles=flags.n_vehicles,
            av_penetration=flags.av_penetration,
            bn_coeff=flags.bn_coeff,
            v_init=flags.v_init,
            c1=flags.c1,
            c2=flags.c2,
            th_target=flags.th_target,
            sigma=flags.sigma,
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
            info[key] = float(info[key])
            print(key, info[key])

    # Generate emission files.
    env.gen_emission(flags.network_type, dir_name)

    # Save statistics.
    del info["expert_action"]
    with open(os.path.join(dir_name, "metrics.json"), "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main(sys.argv[1:])
