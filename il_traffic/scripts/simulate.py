"""Run the expert policies.

This script tests the performance of the Intelligent Driver Model and variants
of the FollowerStopper on the highway and I-210 networks, and collects expert
trajectory data for later use.

Usage
    python simulate.py --network_type "i210" --controller_type 1
"""
import pandas as pd
import sys
import os
import argparse
import json
from copy import deepcopy

from flow.core.util import ensure_dir

from il_traffic.utils.flow_utils import get_network_params
from il_traffic.utils.flow_utils import get_base_env_params
from il_traffic.utils.flow_utils import create_env
from il_traffic.utils.flow_utils import get_emission_path
from il_traffic.utils.visualize import process_emission
from il_traffic.utils.visualize import get_global_position
from il_traffic.utils.visualize import time_space_diagram
from il_traffic.utils.visualize import avg_speed_plot
from il_traffic.utils.visualize import plot_mpg

# additional parameters required for the highway/I-210 networks
HIGHWAY_PARAMS = dict(
    # the inflow rate of vehicles (human and automated)
    inflow=2000,
    # the maximum speed at the downstream boundary edge
    end_speed=5.0,
    # penetration rate of the AVs. 0.10 corresponds to 10%
    penetration_rate=0.05,
)


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
        default='i210',
        help='the type of network to simulate. Must be one of {"highway", '
             '"i210"}.')
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
        '--controller_type',
        type=int,
        default=1,
        help='the type of controller, must be one of: '
             '0) Intelligent Driver Model, '
             '1) FollowerStopper, '
             '2) PISaturation, '
             '3) TimeHeadwayFollowerStopper.')
    parser.add_argument(
        '--noise',
        type=float,
        default=0.0,
        help='the standard deviation of noise assigned to accelerations by '
             'the AVs.')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='whether to print relevant logging data')
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


def rollout(env, model, save_path):
    """Run the rollout, and collect expert samples.

    Parameters
    ----------
    env : il_traffic.ControllerEnv
        the environment to run
    model : function
        the mapping from states to actions by the vehicles. If the model
        returns None, the expert model in the environment is used instead.
    save_path : str
        the folder to save the trajectory data in

    Returns
    -------
    list of float
        list of MPG values collected during the rollout procedure
    list of float
        list of times each MPG value was collected
    """
    obs = env.reset()
    for t in range(env.env_params.horizon):
        # Get the expert action, if needed.
        action = model(obs)

        # Advance the simulation by one step.
        obs, rew, done, _ = env.step(action)

        if done and t < env.env_params.horizon - 1:
            print("Collision occurred. Breaking.")
            break

    # Terminate the environment, and optionally generate emission data.
    mpg_vals = deepcopy(env.mpg_vals)
    mpg_times = deepcopy(env.mpg_times)
    travel_times = deepcopy(env.travel_times)
    env.terminate()

    # Save the travel time data.
    if save_path is not None:
        with open(os.path.join(save_path, "tt.json"), "w") as outfile:
            json.dump({
                "tt": travel_times,
            }, outfile, sort_keys=True, indent=4)

    return mpg_vals, mpg_times


def plot_results(emission_path,
                 network_type,
                 mpg_vals,
                 mpg_times,
                 t_total,
                 use_warmup):
    """Process and plot the data from the trajectory and emission data.

    Parameters
    ----------
    emission_path : str
        the path to the emission.csv file
    network_type : str
        the type of network to simulate. Must be one of {"highway", "i210"}.
    mpg_vals : list of float
        list of MPG values collected during the rollout procedure
    mpg_times : list of float
        list of times each MPG value was collected
    t_total : float
        the total simulation time, in seconds
    use_warmup : bool
        specifies whether to use a warmup file when initializing a simulation
    """
    # Process the emission file.
    process_emission(emission_path, verbose=False)

    # Import the emission file (to be used by the following methods).
    df = pd.read_csv(os.path.join(emission_path, "emission.csv"))

    # Append the global positions.
    df = get_global_position(df, network_type=network_type)

    # Collect some relevant information.
    if network_type == "highway":
        max_speed = 20
        domain_bounds = [500, 2300]
        num_lanes = 1
        observed_min = 500
        observed_max = 2300
        start = 3600
    elif network_type == "i210":
        max_speed = 20
        domain_bounds = [573.08, 2363.27]
        num_lanes = 5
        observed_min = 573.08
        observed_max = 2363.27
        start = 3600
    else:
        # some defaults
        max_speed = 8
        domain_bounds = None
        num_lanes = 1
        observed_min = None
        observed_max = None
        start = 1800

    # Plot the time-space diagram.
    for lane in range(num_lanes):
        time_space_diagram(
            df if use_warmup else [df.time >= (start - 50)],
            lane=lane,
            max_speed=max_speed,
            domain_bounds=domain_bounds,
            title="",
            save_path=os.path.join(emission_path, "ts-{}").format(lane),
            start=0 if use_warmup else start,
        )

    # Plot the average speed.
    avg_speed_plot(
        df,
        min_speed=0,
        max_speed=30,
        title="",
        save_path=os.path.join(emission_path, "avg-speed.png"),
        observed_min=observed_min,
        observed_max=observed_max,
        t_control=None if use_warmup else start,
    )

    # Plot the energy consumption.
    plot_mpg(
        mpg_vals,
        mpg_times,
        t_final=t_total,
        title="",
        max_mpg=60,
        save_path=os.path.join(emission_path, "mpg"),
        t_control=None if use_warmup else start,
    )


def main(args):
    """Run the simulation of the expert policy."""
    # Parse command-line arguments.
    flags = parse_args(args)

    # Get the network parameters.
    network_params = get_network_params(
        inflow=flags.inflow,
        end_speed=flags.end_speed,
        penetration_rate=flags.penetration_rate,
    )

    # Specify the parameters necessary to properly control the automated
    # vehicles.
    environment_params = get_base_env_params(
        network_type=flags.network_type,
        network_params=network_params,
        controller_type=flags.controller_type,
        noise=flags.noise,
        verbose=flags.verbose,
        save_video=flags.save_video,
    )

    # Create the emission directory.
    emission_path = get_emission_path(
        controller_type=flags.controller_type,
        network_type=flags.network_type,
        network_params=network_params,
    )
    ensure_dir(emission_path)

    # Create the environment.
    env = create_env(
        network_type=flags.network_type,
        network_params=network_params,
        environment_params=environment_params,
        render=flags.render,
        emission_path=emission_path,
        use_warmup=flags.use_warmup,
    )

    # Create the expert model.
    def model(_):  # expert defined in the environment
        return None

    # Set the end speed if using the I-210 network.
    if flags.network_type == "i210":
        env.k.kernel_api.edge.setMaxSpeed("119257908#3", flags.end_speed)

    # Run the rollout, and collect expert samples.
    mpg_val, mpg_time = rollout(env=env, model=model, save_path=emission_path)

    # Plot the results from the simulation.
    if flags.gen_emission:
        plot_results(
            emission_path=emission_path,
            network_type=flags.network_type,
            mpg_vals=mpg_val,
            mpg_times=mpg_time,
            t_total=(env.env_params.horizon +
                     env.env_params.warmup_steps) * env.sim_step,
            use_warmup=flags.use_warmup,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
