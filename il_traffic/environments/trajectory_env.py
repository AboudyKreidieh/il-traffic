import time
import bisect
import random
import gym
import numpy as np
import pandas as pd
import math
import os
import json
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

import trajectory.config as t_config

VEHICLE_LENGTH = 5
NUM_VEHICLES = 150
AV_PENETRATION = 0.04


def get_trajectory_from_path(fp, dt):
    df = pd.read_csv(fp)
    v_leader = np.array(df.Velocity / 3.6)

    a_leader = [0.] + list((v_leader[1:] - v_leader[:-1]) / dt)
    v_leader = list(v_leader)
    x_leader = [0.]
    for t in range(1, len(a_leader)):
        x_leader.append(
            x_leader[t-1] 
            + v_leader[t-1] * dt
            + 0.5 * a_leader[t-1] * dt ** 2)

    return x_leader, v_leader, a_leader


def get_inrix_from_dir(directory):
    # Load segment positions.
    with open(os.path.join(directory, "segments.json"), "r") as f:
        x_seg = json.load(f)

    # Load available traffic-state estimation data.
    v_seg = np.genfromtxt(
        os.path.join(directory, "speed.csv"), 
        delimiter=",", skip_header=1)[:, 1:]

    # Import times when the traffic state estimate is updated.
    t_seg = sorted(list(pd.read_csv(
        os.path.join(directory, "speed.csv"))["time"]))
    
    return x_seg, v_seg, t_seg


def get_leader_trajectory(x_seg, v_seg, dt):
    x_leader = [0.]
    v_leader = [get_leader_speed(0., x_seg, v_seg)]
    a_leader = [0.]

    while True:
        x = x_leader[-1]
        v = v_leader[-1]
        a = a_leader[-1]

        x_tp1 = x + v*dt + 0.5*a*dt**2

        if x_tp1 > x_seg[-1]:
            break

        v_tp1 = get_leader_speed(x_tp1, x_seg, v_seg)
        a_tp1 = (v_tp1 - v) / dt

        x_leader.append(x_tp1)
        v_leader.append(v_tp1)
        a_leader.append(a_tp1)

    return x_leader, v_leader, a_leader


def get_leader_speed(x, x_seg, v_seg):
    ix = bisect.bisect(x_seg, x)

    vp = v_seg[ix]
    vm = v_seg[ix - 1]
    xp = x_seg[ix]
    xm = x_seg[ix - 1]

    return vm + (vp - vm) * (x - xm) / (xp - xm)


def get_idm_trajectory(x_leader, v_leader, dt, delay):
    # Choose an initial position at the equilibrium.
    v_eq = v_leader[0]
    h_eq = get_h_eq(v=v_eq, vl=v_leader[0])

    # Set initial positions.
    a_ego = []
    v_ego = [v_eq]
    x_ego = [x_leader[0] - h_eq]

    # Update states.
    horizon = len(x_leader)
    for t in range(1, horizon):
        t_delay = max(0, t - delay)
        x = x_ego[-1]
        v = v_ego[-1]
        vl = v_leader[t_delay]
        h = x_leader[t_delay] - x_ego[-1] - VEHICLE_LENGTH

        a_tp1 = get_idm_accel(v, vl, h, dt)
        v_tp1 = v + a_tp1*dt
        x_tp1 = x + v*dt + 0.5*a_tp1*dt**2

        a_ego.append(a_tp1)
        v_ego.append(v_tp1)
        x_ego.append(x_tp1)

    return x_ego, v_ego, a_ego


def get_h_eq(v, vl):
    v0 = 45
    T = 1.
    a = 1.3
    b = 2.0
    delta = 4
    s0 = 2

    s_star = s0 + max(0, v*T + v*(v-vl) / (2*math.sqrt(a*b)))

    return s_star / math.sqrt(1 - (v/v0)**delta)


def get_idm_accel(v, vl, h, dt):
    v0 = 45
    T = 1.
    a = 1.3
    b = 2.0
    delta = 4
    s0 = 2
    noise = 0.  # 0.3

    s_star = s0 + max(0, v*T + v*(v-vl) / (2*math.sqrt(a*b)))

    accel = a * (1 - (v/v0)**delta - (s_star/h)**2)

    if noise > 0:
        accel += math.sqrt(dt) * random.gauss(0, noise)

    # Make sure speed is not negative.
    accel = max(-v/dt, accel)

    return accel


class NonLocalTrafficFLowHarmonizer(object):
    """A controller that attempts to harmonize traffic near downstream speeds.

    See: TODO
    """

    def __init__(self, dt):
        """Instantiate the vehicle class."""
        self.dt = dt
        self.accel = 0.

        # =================================================================== #
        #                         tunable parameters                          #
        # =================================================================== #

        # estimation type. One of: {"uniform", "gaussian"}
        self._estimation_type = "uniform"
        # scaling term for the response to the proportional error
        self.c1 = 2.0
        # scaling term for the response to the differential error
        self.c2 = 0.5
        # target time headway, in seconds
        self.th_target = 2.
        # standard deviation for the Gaussian smoothing kernel, in meters.
        # For uniform smoothing, this acts as the smoothing window.
        self.sigma = 3000.

        # =================================================================== #
        #                   lower-level control parameters                    #
        # =================================================================== #

        # whether to use the step response approach. If set to False,
        # instantaneous accelerations are computed based on the difference in
        # current and target and smoothed using exponential decay
        self._use_step_response = False

        # second order response parameters for accelerations
        tr1 = 1.6  # rising time, in seconds
        pos1 = 0.05  # percent overshoot
        zeta = np.sqrt(np.log(pos1) ** 2 / (np.pi ** 2 + np.log(pos1) ** 2))
        phi = np.arctan2(np.sqrt(1 - zeta ** 2), zeta)
        omega_n = (np.pi - phi) / (tr1 * np.sqrt(1 - zeta ** 2))
        self._zeta1 = zeta
        self._omega_n1 = omega_n

        # second order response parameters for decelerations
        tr2 = 0.8  # rising time, in seconds
        pos2 = 0.11  # percent overshoot
        zeta = np.sqrt(np.log(pos2) ** 2 / (np.pi ** 2 + np.log(pos2) ** 2))
        phi = np.arctan2(np.sqrt(1 - zeta ** 2), zeta)
        omega_n = (np.pi - phi) / (tr2 * np.sqrt(1 - zeta ** 2))
        self._zeta2 = zeta
        self._omega_n2 = omega_n

        # two most recent accelerations, in m/s^2
        self._a_tm1 = 0.
        self._a_tm2 = 0.
        # largest possible acceleration, in m/s^2
        self.max_accel = 1.5
        # largest possible deceleration, in m/s^2
        self.max_decel = 3.0
        # exponential decay component for change in accelerations
        self.gamma = 0.5

        # =================================================================== #
        #                         failsafe parameters                         #
        # =================================================================== #

        # prediction time horizon, in seconds
        self.tau = 5.0
        # minimum allowed space headway, in meters
        self.h_min = 5
        # minimum allowed time headway, in seconds
        self.th_min = 0.5

        # =================================================================== #
        #                          other parameters                           #
        # =================================================================== #

        # current target speed, in m/s
        self.v_des = 30.
        # largest assignable target speed, in m/s
        self.v_max = 40.
        # buffer of leading vehicle speeds
        self._vl = deque(maxlen=100)
        self._lead_vel = 30.

    def reset(self):
        """TODO."""
        self.accel = 0.
        self.v_des = 30.
        self._vl.clear()

    @staticmethod
    def _gaussian(x0, x, z, sigma):
        """Perform a kernel smoothing operation on future average speeds."""
        # Collect relevant traffic-state info.
        # - ix0: segments in front of the ego vehicle
        # - ix1: final segment, or clipped to estimate congested speeds
        ix0 = next(i for i in range(len(x)) if x[i] >= x0)
        try:
            ix1 = next(j for j in range(ix0 + 2, len(x)) if
                       z[j] - np.mean(z[ix0:j]) > 15) + 1
        except StopIteration:
            ix1 = len(x)
        x = x[ix0:ix1]
        z = z[ix0:ix1]

        densities = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -np.square(x - x0) / (2 * sigma ** 2))

        densities = densities / sum(densities)

        return sum(densities * z)

    @staticmethod
    def _uniform(x0, x, z, width):
        """TODO."""
        # Collect relevant traffic-state info.
        # - ix0: segments in front of the ego vehicle
        # - ix1: final segment, or clipped to estimate congested speeds
        ix0 = next(i for i in range(len(x)) if x[i] >= x0)
        try:
            ix1 = next(j for j in range(ix0 + 2, len(x))
                       if x[j] >= (x0 + width)
                       or z[j] - np.mean(z[ix0:j]) > 15) + 1
        except StopIteration:
            ix1 = len(x)

        x = np.array(deepcopy(x[ix0-1:ix1]))
        z = np.array(deepcopy(z[ix0-1:ix1]))

        # Replace endpoints with a cutoff
        z[0] = z[0] + (z[1]-z[0]) * (x0-x[0]) / (x[1]-x[0])
        x[0] = x0

        if x[-1] > x0 + width:
            z[-1] = z[-2] + (z[-1]-z[-2]) * (x0+width-x[-2]) / (x[-1]-x[-2])
            x[-1] = x0 + width

        area = 0.5 * (z[1:]+z[:-1]) * (x[1:]-x[:-1])
        actual_width = x[-1] - x[0]

        return sum(area) / actual_width

    def _get_accel_from_vdes(self, v_t, v_des):
        """Return the desired acceleration."""
        if self._use_step_response:
            # Apply step response.
            if v_des - v_t > -0.25:
                accel = self._2or(v_t, v_des, self._omega_n1, self._zeta1)
            elif v_des - v_t <= -0.25:
                accel = self._2or(v_t, v_des, self._omega_n2, self._zeta2)
            else:
                accel = 0.
        else:
            # Apply instantaneous smoothed acceleration.
            accel = self._bounded_accel(v_t=v_t, v_des=v_des)

        return accel

    def _bounded_accel(self, v_t, v_des):
        """Return a bounded acceleration to instantaneously achieve target."""
        # Compute bounded acceleration.
        accel = max(
            -abs(self.max_decel), min(self.max_accel, (v_des-v_t)/self.dt))

        # Reduce noise.
        accel = self.gamma * accel + (1. - self.gamma) * self.accel

        return accel

    def _2or(self, v_t, v_des, omega_n, zeta):
        """Update acceleration based on second order response."""
        y_tm1 = self._a_tm1
        y_tm2 = self._a_tm2
        x = v_des - v_t  # TODO: I think
        dt = self.dt

        # Compute next acceleration.
        y_t = 2 * (1 - zeta * omega_n * dt) * y_tm1 \
            - (1 - 2 * omega_n * dt + (omega_n * dt) ** 2) * y_tm2 \
            + x

        # Update memory.
        self._a_tm2 = self._a_tm1
        self._a_tm1 = y_t

        return y_t

    def get_accel(self, v, vl, h, x, x_seg, v_seg):
        # Update the buffer of lead speeds.
        self._vl.append(vl)
        self._lead_vel = 0.9 * self._lead_vel + 0.1 * vl

        if x_seg is not None:
            # Compute time headway.
            th = h / (v + 1e-6)
            th = max(0., min(40., th))
            th_error = th - self.th_target
            delta_v = self._lead_vel - v

            # weighting between CACC and ACC
            alpha = max(0., min(1., th - 1))

            # Choose downstream estimation method.
            estimation_method = \
                self._uniform if self._estimation_type == "uniform" \
                else self._gaussian

            # Compute target speed.
            v_target = \
                alpha * estimation_method(x, x_seg, v_seg, self.sigma) + \
                (1. - alpha) * self.v_des + \
                self.c1 * th_error + \
                self.c2 * delta_v

            # Update desired speed.
            max_decel = -1.0
            max_accel = 1.0
            self.v_des = max(
                self.v_des + max_decel * self.dt, min(
                    self.v_des + max_accel * self.dt,
                    v_target,
                )
            )

        # Predict the average future acceleration for the leader.
        if len(self._vl) > 1:
            ix = min(len(self._vl), 50)  # 5 seconds
            _vl = list(self._vl)[-int(ix):]
            a_lead = max(0., (_vl[-1] - _vl[0]) / (self.dt*(len(_vl) - 1)))
        else:
            a_lead = 0.

        # Clip by values that maintain a safe time headway.
        self.v_des = max(0., min(
            self.v_des,
            (h
             - self.h_min
             + vl * self.tau
             + 0.5 * a_lead * self.tau ** 2
             - 0.5 * v * self.tau) / (self.th_min + 0.5 * self.tau)
        ))

        # Compute desired acceleration.
        self.accel = self._get_accel_from_vdes(v_t=v, v_des=self.v_des)

        return self.accel


class TrajectoryEnv(gym.Env):

    def __init__(self):
        # simulation step size
        self.dt = 0.1
        # human time delay, in steps
        self.delay = 0
        # platoon index
        self._platoon_ix = 0
        # the names of all valid trajectories for training purposes
        self.fp = [
            '2021-03-09-13-35-04_2T3MWRFVXLW056972_masterArray_0_6825',
            '2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_0_4917',
            '2021-03-15-12-46-38_2T3MWRFVXLW056972_masterArray_1_11342',
            '2021-03-24-12-39-15_2T3MWRFVXLW056972_masterArray_0_6438',
            '2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_0_11294',
            '2021-04-07-12-33-03_2T3MWRFVXLW056972_masterArray_1_6116',
            '2021-04-16-12-34-41_2T3MWRFVXLW056972_masterArray_0_5778',
            '2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_0_16467',
            '2021-04-19-12-27-33_2T3MWRFVXLW056972_masterArray_1_6483',
            '2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050',
        ]

        # expert controller
        self.expert = NonLocalTrafficFLowHarmonizer(dt=self.dt)

        # acceleration memory
        self.prev_accel = 0.

        self.x = []
        self.v = []
        self.a = []
        self.all_vdes = []

        self.x_seg = []
        self.v_seg = []
        self.t_seg = []

    def reset(self, _add_av=True):
        fp = random.sample(self.fp, 1)[0]
        directory = os.path.join(
            t_config.PROJECT_PATH,
            'dataset/data_v2_preprocessed_west/{}'.format(fp))

        # Add leader trajectory.
        x_leader, v_leader, a_leader = get_trajectory_from_path(
            fp=os.path.join(directory, "trajectory.csv"), dt=self.dt)

        self.x = [x_leader]
        self.v = [v_leader]
        self.a = [a_leader]
        self.all_vdes = []

        # Get segment data.
        x_seg, v_seg, t_seg = get_inrix_from_dir(directory)

        self.x_seg = x_seg
        self.v_seg = v_seg
        self.t_seg = t_seg

        # Reset platoon index.
        self._platoon_ix = 0

        if _add_av:
            # Place an automated vehicle.
            v_eq = v_leader[0]
            h_eq = get_h_eq(v=v_eq, vl=v_leader[0])

            self.a.append([])
            self.v.append([v_eq])
            self.x.append([x_leader[0] - h_eq])
            self.all_vdes.append([])

            # Reset expert controller and memory.
            self.expert.reset()
            self.prev_accel = 0.

            state = self.get_state(
                x_av=self.x[-1],
                v_av=self.v[-1],
                x_leader=self.x[-2],
                v_leader=self.v[-2],
            )
        else:
            state = []

        return state

    def step(self, action):
        """See parent class."""
        done = False

        # ego vehicle state
        x_av = self.x[-1]
        v_av = self.v[-1]

        # lead vehicle state
        x_leader = self.x[-2]
        v_leader = self.v[-2]

        # time and segment index
        t = len(x_av) - 1
        ix_t = bisect.bisect(self.t_seg, self.dt * t) - 1

        # Compute expert action.
        expert_action = self.expert.get_accel(
            v=v_av[t],
            vl=v_leader[t],
            h=x_leader[t] - x_av[t] - VEHICLE_LENGTH,
            x=x_av[t],
            x_seg=self.x_seg,
            v_seg=self.v_seg[ix_t],
        )
        expert_vdes = self.expert.v_des
        self.all_vdes[-1].append(expert_vdes)

        # Get AV's acceleration.
        accel = expert_action if action is None else self.get_av_accel(action)

        # Update dynamics.
        x_tp1 = x_av[-1] + v_av[-1] * self.dt + 0.5 * accel * self.dt ** 2
        v_tp1 = v_av[-1] + accel * self.dt

        self.x[-1].append(x_tp1)
        self.v[-1].append(v_tp1)
        self.a[-1].append(accel)

        # Check if automated vehicle is done computing its actions.
        if len(self.v[-1]) == len(self.v[0]):
            self._platoon_ix += 1
            print(self._platoon_ix)

            if self._platoon_ix >= NUM_VEHICLES:
                # Check if done conditions were met.
                done = True
            else:
                # Add human vehicles.
                while self._platoon_ix % int(1/AV_PENETRATION) != 0:
                    # Increment platoon index.
                    self._platoon_ix += 1

                    # Compute human-driven dynamics.
                    x_ego, v_ego, a_ego = get_idm_trajectory(
                        x_leader=self.x[-1],
                        v_leader=self.v[-1],
                        dt=self.dt,
                        delay=self.delay,
                    )

                    # Append vehicle data.
                    self.x.append(x_ego)
                    self.v.append(v_ego)
                    self.a.append(a_ego)

                # Initialize next automated vehicle.
                x_leader = self.x[-1]
                v_leader = self.v[-1]
                v_eq = v_leader[0]
                h_eq = get_h_eq(v=v_eq, vl=v_leader[0])

                self.a.append([])
                self.v.append([v_eq])
                self.x.append([x_leader[0] - h_eq - VEHICLE_LENGTH])
                self.all_vdes.append([])

                # Reset expert controller and memory.
                self.expert.reset()
                self.prev_accel = 0.

        # Check for a collision
        h_t = self.x[-2][len(self.x[-1]) - 1] - self.x[-1][-1] - VEHICLE_LENGTH
        if h_t <= 0:
            print("Collision")
            done = True

        # Return state and expert action.
        next_obs = self.get_state(
            x_av=self.x[-1],
            v_av=self.v[-1],
            x_leader=self.x[-2],
            v_leader=self.v[-2],
        )
        reward = 0.
        info = {"expert_actions": [np.array([expert_vdes])]}

        return next_obs, reward, done, info

    def get_av_accel(self, action):
        """Convert actions to a desired acceleration."""
        v_des = action[0][0]

        # failsafe parameters
        h_min = 5.
        tau = 5.
        th_min = 0.5

        t = len(self.x[-1]) - 1
        x_t = self.x[-1][-1]
        v_t = self.v[-1][-1]
        xl_t = self.x[-2][t]
        vl_t = self.v[-2][t]
        h_t = xl_t - x_t - VEHICLE_LENGTH

        # approximation for leader acceleration
        t0 = max(0, t - 50)
        a_lead = (self.v[t] - self.v[t0]) / (self.dt * (t - t0))

        # Update desired speed based on safety.
        v_des = max(0., min(
            v_des,
            (h_t
             - h_min
             + vl_t * tau
             + 0.5 * a_lead * tau ** 2
             - 0.5 * v_t * tau) / (th_min + 0.5 * tau)
        ))

        # Compute bounded acceleration.
        accel = max(-3.0, min(1.5, (v_des - v_t) / self.dt))

        # Reduce noise.
        gamma = 0.5
        accel = gamma * accel + (1. - gamma) * self.prev_accel

        self.prev_accel = accel

        return accel

    @staticmethod
    def get_state(x_av, v_av, x_leader, v_leader):
        """Return the agent's observation.

        This observation consists of:

        1. Ego time headway
        2. Ego speed
        3. Previous desired speed
        4. History of leader's speeds
        """
        history_length = 50
        speed_scale = 40.
        th_scale = 10.

        t = len(x_av)
        t0 = max(0, t - history_length)

        x_t = x_av[-1]
        v_t = v_av[-1]
        h_t = x_leader[t-1] - x_t - VEHICLE_LENGTH
        th_t = h_t / v_t

        obs = np.array(
            [th_t / th_scale, v_t / speed_scale]
            + [0.] * (50 - t + t0)
            + [val / speed_scale for val in v_leader[t0:t]])

        return [obs]

    def get_automated_trajectory(self, x_leader, v_leader):
        # Create a controller object.
        veh = NonLocalTrafficFLowHarmonizer(dt=self.dt)

        # Choose an initial position at the equilibrium.
        v_eq = v_leader[0]
        h_eq = get_h_eq(v=v_eq, vl=v_leader[0])

        # Set initial positions.
        a_av = []
        v_av = [v_eq]
        x_av = [x_leader[0] - h_eq]
        v_des = []

        for t in range(len(x_leader) - 1):
            ix_t = bisect.bisect(self.t_seg, self.dt * t) - 1

            # Compute desired acceleration by automated vehicle.
            accel = veh.get_accel(
                v=v_av[t],
                vl=v_leader[t],
                h=x_leader[t] - x_av[t] - VEHICLE_LENGTH,
                x=x_av[t],
                x_seg=self.x_seg,
                v_seg=self.v_seg[ix_t],
            )
            v_des.append(veh.v_des)

            # Update dynamics.
            x_av.append(x_av[-1] + v_av[-1]*self.dt + 0.5*accel*self.dt**2)
            v_av.append(v_av[-1] + accel*self.dt)
            a_av.append(accel)

        return x_av, v_av, a_av, v_des

    def plot_statistics(self):
        x = self.x
        v = self.v
        v_des = self.all_vdes
        times = list(np.arange(len(x[0])) * self.dt)

        plt.figure(figsize=(16, 4))
        for i in range(1, len(x), 10):
            plt.plot(x[i], v[i])
        plt.plot(x[0], v[0], c="k", lw=2, label="leader")
        plt.legend(fontsize=15)
        plt.xlim([0, np.max(x)])
        plt.ylim([-0.025 * np.max(v), 1.025 * np.max(v)])
        plt.grid(linestyle='--')
        plt.xticks(fontsize=15)
        plt.xlabel("Position (m)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Speed (m/s)", fontsize=15)
        plt.show()

        plt.figure(figsize=(16, 4))
        for i in range(1, len(x), 10):
            plt.plot(times, v[i])
        plt.plot(times, v[0], c="k", lw=2, label="leader")
        plt.legend(fontsize=15)
        plt.ylim([-0.025 * np.max(v), 1.025 * np.max(v)])
        plt.grid(linestyle='--')
        plt.xticks(fontsize=15)
        plt.xlabel("Time (s)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Speed (m/s)", fontsize=15)
        plt.show()

        plt.figure(figsize=(16, 4))
        times = list(np.arange(len(x[0])) * self.dt)
        for i in range(0, len(x), 10):
            plt.plot(times, np.array(x[i]) / 1000, c='k')
        plt.grid(linestyle='--')
        plt.xticks(fontsize=15)
        plt.xlabel("Time (s)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Position (km)", fontsize=15)
        plt.show()

        plt.figure(figsize=(16, 4))
        for i in range(len(v_des)):
            plt.plot(v_des[i], c='k')
        plt.grid(linestyle='--')
        plt.xticks(fontsize=15)
        plt.xlabel("Time (s)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Desired speed (m/s)", fontsize=15)
        plt.show()

    def rollout(self):
        t0 = time.time()

        _ = self.reset(_add_av=False)

        while self._platoon_ix < NUM_VEHICLES:
            if self.x[-1][-1] < 0:
                print(self._platoon_ix)
                break

            # Update automated vehicle trajectory after initial
            # position of the leader.
            if AV_PENETRATION > 0 \
                    and self._platoon_ix % int(1/AV_PENETRATION) == 0:
                print(self._platoon_ix)
                x_ego, v_ego, a_ego, v_des_i = self.get_automated_trajectory(
                    x_leader=self.x[-1],
                    v_leader=self.v[-1],
                )
                self.all_vdes.append(v_des_i)
            else:
                # Compute human-driven dynamics.
                x_ego, v_ego, a_ego = get_idm_trajectory(
                    x_leader=self.x[-1],
                    v_leader=self.v[-1],
                    dt=self.dt,
                    delay=self.delay,
                )

            # Append vehicle data.
            self.x.append(x_ego)
            self.v.append(v_ego)
            self.a.append(a_ego)

            if any(self.x[-2][t] - self.x[-1][t] < VEHICLE_LENGTH
                   for t in range(len(self.x[-1]))):
                print("Collision")
                break

            # Increment platoon index.
            self._platoon_ix += 1

        print(time.time() - t0)

        return self.x, self.v, self.a, self.all_vdes

    def render(self, mode="human"):
        """See parent class."""
        pass
