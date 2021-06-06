"""Auxiliary visualization methods."""
import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection


# These edges have an extra lane that RL vehicles do not traverse (since they
# do not change lanes). We as a result ignore their first lane computing per-
# lane observations.
EXTRA_LANE_EDGES = [
    "119257908#1-AddedOnRampEdge",
    "119257908#1-AddedOffRampEdge",
    ":119257908#1-AddedOnRampNode_0",
    ":119257908#1-AddedOffRampNode_0",
    "119257908#3",
]


def process_emission(emission_path, verbose):
    """Process the generated emission file.

    This method performs two operations:

    1. It renames the csv file to "emission.csv".
    2. It reduces the precision of a few columns to save space.

    Parameters
    ----------
    emission_path : str
        the path to the folder containing the emission file
    verbose : bool
        whether to comment of operations running
    """
    if verbose:
        print("Renaming emission file.")

    # Get the name of the emission file.
    csv_file = [x for x in os.listdir(emission_path) if x.endswith("csv")][0]
    original_fp = os.path.join(emission_path, csv_file)

    # Rename it to "emission.csv".
    new_fp = os.path.join(emission_path, "emission.csv")
    os.rename(original_fp, new_fp)

    if verbose:
        print("Shrinking emission file.")

    # Reduce to 3-decimal points.
    df = pd.read_csv(new_fp)
    df = df.round(3).astype(str)
    df = df.replace(to_replace="\.0+$", value="", regex=True)  # noqa: W605
    df = df.replace(to_replace="nan", value="", regex=True)

    # Keep only relevant columns.
    df[['time',
        'id',
        'type',
        'speed',
        'headway',
        'target_accel_with_noise_with_failsafe',
        'target_accel_no_noise_no_failsafe',
        'target_accel_with_noise_no_failsafe',
        'target_accel_no_noise_with_failsafe',
        'edge_id',
        'lane_number',
        'relative_position']].to_csv(new_fp, index=False)

    # Clear memory.
    del df


def get_global_position(df, network_type):
    """Append the global position of vehicles to the emission dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        the original emission dataframe
    network_type : str
        the type of network to simulate. Must be one of {"highway", "i210"}.

    Returns
    -------
    pd.DataFrame
        the emission dataframe with the global positions
    """
    if network_type == "highway":
        highway_length = 2500
        edgestarts = {
            'highway_0': 0,
            ':edge_1_0': highway_length / 2,
            'highway_1': highway_length / 2 + 0.1,
            ':edge_2_0': highway_length + 0.1,
            'highway_end': highway_length + 0.2,
        }

    elif network_type == "i210":
        edgestarts = {
            'ghost0': 0,
            ':300944378_0': 573.08,
            '119257914': 573.38,
            ':300944379_0': 634.66,
            '119257908#0': 634.9699999999999,
            ':300944436_0': 1331.94,
            '119257908#1-AddedOnRampEdge': 1334.81,
            ':119257908#1-AddedOnRampNode_0': 1432.01,
            '119257908#1': 1435.25,
            ':119257908#1-AddedOffRampNode_0': 1674.93,
            '119257908#1-AddedOffRampEdge': 1678.17,
            ':1686591010_1': 1776.67,
            '119257908#2': 1782.13,
            ':1842086610_1': 2358.7400000000002,
            '119257908#3': 2363.2700000000004
        }

    else:
        raise ValueError("Unknown network type: {}".format(network_type))

    # create a column for the global positions
    df["global_position"] = 0

    # add the global position of the start of the edge
    for key in edgestarts.keys():
        df.loc[df.edge_id == key, "global_position"] = edgestarts[key]

    # add the relative position of the vehicles on the edge
    df.global_position += df.relative_position

    return df


def time_space_diagram(df,
                       lane=0,
                       domain_bounds=None,
                       start=None,
                       min_speed=0,
                       max_speed=30,
                       discontinuity=200,
                       title="Time-Space Diagram",
                       save_path=None):
    """Plot the time-space diagram.

    Parameters
    ----------
    df : pd.DataFrame
        the emission dataframe
    lane : int
        the lane to draw the time-space diagram for
    domain_bounds : [float, float] or None
        the [pos_min, pos_max] bounds of the positions of the controllable
        region. The positions outside this in the plot are greyed out.
    start : float
        the start time of control. The times outside before this in the plot
        are greyed out.
    min_speed : float
        minimum speed for the color-bar
    max_speed : float
        maximum speed for the color-bar
    discontinuity : float
        a value to account for loop-overs in ring networks. If two numbers are
        offset by this value, no line is drawn between them.
    title : str
        the title for the plot. If set to "", no title is added.
    save_path : str
        the name of the file to save
    """
    # leave only lane-relevant data
    df = df[
        (df.edge_id.isin(EXTRA_LANE_EDGES) & (df.lane_number == (lane + 1))) |
        ((~df.edge_id.isin(EXTRA_LANE_EDGES)) & (df.lane_number == lane))
    ]

    # extract some variables from the dataset
    times = sorted(list(np.unique(df.time)))
    num_times = len(times)
    indx_times = {str(round(k, 1)): i for i, k in enumerate(times)}
    vehicles = sorted(list(np.unique(df.id)))
    num_vehicles = len(vehicles)
    indx_vehicles = {k: i for i, k in enumerate(vehicles)}

    # empty arrays for the positions and speeds of all vehicles
    pos = - np.ones((num_times, num_vehicles))
    speed = - np.ones((num_times, num_vehicles))

    # prepare the speed and absolute position in a way that is compatible with
    # the space-time diagram, and compute the number of vehicles at each step
    for i in df.index:
        veh_id = df.at[i, "id"]
        pos_i = df.at[i, "global_position"]
        speed_i = df.at[i, "speed"]
        time_i = str(df.at[i, "time"])
        pos[indx_times[time_i], indx_vehicles[veh_id]] = pos_i
        speed[indx_times[time_i], indx_vehicles[veh_id]] = speed_i

    # some plotting parameters
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    my_cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    pos = pos[::3, :]
    speed = speed[::3, :]
    times = np.array(times[::3])

    # perform plotting operation
    plt.figure(figsize=(9, 6))
    ax = plt.axes()
    norm = plt.Normalize(min_speed, max_speed)
    cols = []

    xmin = times[0] - (start or 0)
    xmax = times[-1] - (start or 0)
    xbuffer = (xmax - xmin) * 0.025  # 2.5% of range
    ymin, ymax = np.amin(pos), np.amax(pos)
    ybuffer = (ymax - ymin) * 0.025  # 2.5% of range

    ax.set_xlim(xmin - xbuffer, xmax + xbuffer)
    ax.set_ylim(
        ymin - ybuffer - (0 if domain_bounds is None else domain_bounds[0]),
        ymax + ybuffer - (0 if domain_bounds is None else domain_bounds[0]))

    for indx_car in range(pos.shape[1]):
        unique_car_pos = pos[:, indx_car]

        indx = unique_car_pos >= 0
        unique_car_time = times[indx] - (start or 0)
        unique_car_pos = unique_car_pos[indx] - (
            domain_bounds[0] if domain_bounds is not None else 0)
        unique_car_speed = speed[indx, indx_car]

        # discontinuity from wraparound
        disc = np.where(
            np.abs(np.diff(unique_car_pos)) >= discontinuity)[0] + 1
        unique_car_time = np.insert(unique_car_time, disc, np.nan)
        unique_car_pos = np.insert(unique_car_pos, disc, np.nan)
        unique_car_speed = np.insert(unique_car_speed, disc, np.nan)

        points = np.array(
            [unique_car_time, unique_car_pos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=my_cmap, norm=norm)

        # Set the values used for color mapping
        lc.set_array(unique_car_speed)
        lc.set_linewidth(1.75)
        cols.append(lc)

    plt.title(title, fontsize=20)
    plt.ylabel('Position (m)', fontsize=20)
    plt.xlabel('Time (s)', fontsize=20)

    if domain_bounds is not None or start is not None:
        # Make sure a domain bound is specified, and if not assign it.
        domain_bounds = domain_bounds or [ymin, ymax]

        rects = [
            # rectangle for warmup period, but not ghost edges
            Rectangle((xmin, -domain_bounds[0]), 0 - xmin, domain_bounds[1]),
            # rectangle for lower ghost edge (including warmup period)
            Rectangle((0, ymin - domain_bounds[0]), xmax,
                      domain_bounds[0]),
            # rectangle for upper ghost edge (including warmup period)
            Rectangle((xmin, domain_bounds[1] - domain_bounds[0]),
                      xmax - xmin, ymax - domain_bounds[1])
        ]

        pc = PatchCollection(rects, facecolor='grey', alpha=0.5,
                             edgecolor=None)
        pc.set_zorder(20)
        ax.add_collection(pc)

    for col in cols:
        line = ax.add_collection(col)
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('Velocity (m/s)', fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if save_path is not None:
        plt.savefig("{}.png".format(save_path), bbox_inches='tight')
    plt.clf()
    plt.close()


def avg_speed_plot(df,
                   df_compare=None,
                   min_speed=0,
                   max_speed=30,
                   t_min=None,
                   t_max=None,
                   t_control=None,
                   observed_min=None,
                   observed_max=None,
                   title="Average Speed Plot",
                   save_path=None):
    """Plot the average speed of vehicles in a network.

    Parameters
    ----------
    df : pd.DataFrame
        the emission dataframe
    df_compare : pd.DataFrame
        the baseline emission dataframe. If specified, this will be included in
        the plot.
    min_speed : float
        the lower bound of the speed (y-value) in the plot
    max_speed : float
        the upper bound of the speed (y-value) in the plot
    t_min : float
        the lower bound of the time (x-value) in the plot
    t_max : float
        the upper bound of the time (x-value) in the plot
    t_control : float
        the time when control is activated. If specified, a vertical dotted
        line is drawn here.
    observed_min : float
        the lower bound of the positions of vehicles that are included in the
        average speed computations. If set to None, there is no lower bound
    observed_max : float
        the upper bound of the positions of vehicles that are included in the
        average speed computations. If set to None, there is no upper bound
    title : str
        the title for the plot. If set to "", no title is added.
    save_path : str
        the name of the file to save
    """
    # Keep only data for the given time.
    if t_min is not None:
        df = df[df.time >= t_min]
        if df_compare is not None:
            df_compare = df_compare[df_compare.time >= t_min]
    if t_max is not None:
        df = df[df.time <= t_max]
        if df_compare is not None:
            df_compare = df_compare[df_compare.time <= t_max]

    # Keep only data for the given observed positions.
    if observed_min is not None:
        df = df[df.global_position >= observed_min]
        if df_compare is not None:
            df_compare = df_compare[df_compare.global_position >= observed_min]
    if observed_max is not None:
        df = df[df.global_position <= observed_max]
        if df_compare is not None:
            df_compare = df_compare[df_compare.global_position <= observed_max]

    # extract some variables from the dataset
    times = np.array(sorted(list(np.unique(df.time))))
    avg_speeds, std_speeds = [], []
    avg_speeds_compare, std_speeds_compare = [], []
    for time_i in times:
        speeds = df[df.time == time_i].speed
        avg_speeds.append(np.mean(speeds))
        std_speeds.append(np.std(speeds))
        if df_compare is not None:
            speeds_compare = df_compare[df_compare.time == time_i].speed
            avg_speeds_compare.append(np.mean(speeds_compare))
            std_speeds_compare.append(np.std(speeds_compare))
    avg_speeds = np.array(avg_speeds)
    std_speeds = np.array(std_speeds)
    avg_speeds_compare = np.array(avg_speeds_compare)
    std_speeds_compare = np.array(std_speeds_compare)

    # Plot the figure.
    plt.figure(figsize=(9, 6))

    plt.title(title, fontsize=30)
    plt.ylabel('Speed (m/s)', fontsize=25)
    plt.xlabel('Time (s)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(linestyle='dotted', lw=1)

    if min_speed is not None:
        y_range = max_speed - min_speed
        plt.ylim(min_speed - 0.1 * y_range, max_speed + 0.1 * y_range)
    if t_min is not None and t_max is not None:
        plt.xlim(t_min - (t_control or 0), t_max - (t_control or 0))

    if df_compare is not None:
        plt.plot(times - (t_control or 0), avg_speeds_compare, c='orange',
                 lw=2, label="baseline")
        plt.fill_between(
            times - (t_control or 0),
            avg_speeds_compare - std_speeds_compare,
            avg_speeds_compare + std_speeds_compare,
            color='orange', alpha=0.25)

    plt.plot(times - (t_control or 0), avg_speeds, c='b', lw=2,
             label="control")
    plt.fill_between(
        times - (t_control or 0),
        avg_speeds - std_speeds, avg_speeds + std_speeds,
        color='b', alpha=0.25)

    # Add the legend only to the comparison plots.
    if df_compare is not None:
        plt.legend(fontsize=20)

    if t_control is not None:
        plt.axvline(x=0, ls='--', lw=2, c='k')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def plot_mpg(mpg_vals,
             mpg_times,
             t_final,
             max_mpg=100,
             t_control=None,
             title="Miles-per-Gallon Plot",
             save_path=None):
    """Plot the miles-per-gallon of the vehicles in the network.

    Parameters
    ----------
    mpg_vals : list of float
        the mpg values for vehicles in increments of 50 m
    mpg_times : list of float
        the times that the vehicle crossed 50 m
    t_final : float
        the total simulation time (in seconds)
    max_mpg : float
        the large y-limit for the plot. The minimum is set to 0.
    t_control : float
        the time when control is activated. If specified, a vertical dotted
        line is drawn here.
    title : str
        the title for the plot. If set to "", no title is added.
    save_path : str
        the name of the file to save
    """
    # ======================================================================= #
    # Save mpg data.                                                          #
    # ======================================================================= #

    d = {"mpg_vals": mpg_vals, "mpg_times": mpg_times}
    with open("{}.csv".format(save_path), "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))

    # ======================================================================= #
    # Create and save mpg plot.                                               #
    # ======================================================================= #

    mpg_vals = np.array(mpg_vals)
    mpg_times = np.array(mpg_times)

    mpg_times = mpg_times[mpg_vals < max_mpg]
    mpg_vals = mpg_vals[mpg_vals < max_mpg]

    plt.figure(figsize=(9, 6))

    if t_control is not None:
        mpg_times = mpg_times - t_control
        plt.axvline(x=0, ls='--', lw=2, c='k')

    dt = 25
    mpg_increments = []
    for t in np.arange(0 - (t_control or 0), t_final - (t_control or 0), dt):
        mpg_increments.append(
            np.mean(mpg_vals[(t <= mpg_times) & (mpg_times <= t + dt)]))

    plt.title(title, fontsize=30)
    plt.ylabel('Miles per Gallon', fontsize=25)
    plt.xlabel('Time (s)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(linestyle='dotted', lw=1)
    plt.ylim([0, max_mpg])

    plt.scatter(mpg_times, mpg_vals, alpha=0.25, c='b')
    plt.plot(np.arange(0 - (t_control or 0), t_final - (t_control or 0), dt),
             mpg_increments, c='k', lw=3)

    if t_control is not None:
        plt.axvline(x=0, ls='--', lw=2, c='k')

    if save_path is not None:
        plt.savefig("{}.png".format(save_path), bbox_inches='tight')
    plt.clf()
    plt.close()
