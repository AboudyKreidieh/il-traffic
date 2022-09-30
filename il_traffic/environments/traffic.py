import gym
import pandas as pd
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from collections import deque

# energy model parameters
RAV4_2019_COEFFS = {
    'beta0': 0.013111753095302022,
    'vc': 5.98,
    'p1': 0.047436831067050676,
    'C0': 0.14631964767035743,
    'C1': 0.012179045946260292,
    'C2': 0,
    'C3': 2.7432588728174234e-05,
    'p0': 0.04553801347643801,
    'p2': 0.0018022443124799303,
    'q0': 0,
    'q1': 0.02609037187916979,
    'b1': 7.1780386096154185,
    'b2': 0.053537268955100234,
    'b3': 0.27965662935753677,
    'z0': 1.4940081773441736,
    'z1': 1.2718495543500672,
    'ver': '2.0',
    'mass': 1717,
    'fuel_type': 'gasoline',
}
# conversions
GRAMS_PER_SEC_TO_GALS_PER_HOUR = {
    'diesel': 1.119,  # 1.119 gal/hr = 1g/s
    'gasoline': 1.268,  # 1.268 gal/hr = 1g/s
}


def get_idm_coeff(network_type):
    assert network_type in ["i24", "bottleneck"]

    coeff = {
        "v0": 33,
        "T": 1.,
        "a": 1.3,
        "b": 2.0,
        "delta": 4.,
        "s0": 2.,
        "noise": 0.0,  # 0.3
        "vlength": 4.,
    }

    if network_type == "i24":
        coeff["v0"] = 45.
        coeff["vlength"] = 5.

    return coeff


def get_idm_accel(v, vl, h, dt, network_type):
    coeff = get_idm_coeff(network_type)
    v0 = coeff["v0"]
    T = coeff["T"]
    a = coeff["a"]
    b = coeff["b"]
    delta = coeff["delta"]
    s0 = coeff["s0"]
    noise = coeff["noise"]

    s_star = s0 + max(0, v*T + v*(v-vl) / (2*math.sqrt(a*b)))

    accel = a * (1 - (v/v0)**delta - (s_star/h)**2)

    if noise > 0:
        accel += math.sqrt(dt) * random.gauss(0, noise)

    # Make sure speed is not negative.
    accel = max(-v/dt, accel)

    return accel


def get_h_eq(v, vl, network_type):
    coeff = get_idm_coeff(network_type)
    v0 = coeff["v0"]
    T = coeff["T"]
    a = coeff["a"]
    b = coeff["b"]
    delta = coeff["delta"]
    s0 = coeff["s0"]

    s_star = s0 + max(0, v*T + v*(v-vl) / (2*math.sqrt(a*b)))

    return s_star / math.sqrt(1 - (v/v0)**delta)


class PFM2019RAV4(object):

    def __init__(self):
        self.mass = RAV4_2019_COEFFS['mass']
        self.state_coeffs = np.array([RAV4_2019_COEFFS['C0'],
                                      RAV4_2019_COEFFS['C1'],
                                      RAV4_2019_COEFFS['C2'],
                                      RAV4_2019_COEFFS['C3'],
                                      RAV4_2019_COEFFS['p0'],
                                      RAV4_2019_COEFFS['p1'],
                                      RAV4_2019_COEFFS['p2'],
                                      RAV4_2019_COEFFS['q0'],
                                      RAV4_2019_COEFFS['q1'],
                                      RAV4_2019_COEFFS['z0'],
                                      RAV4_2019_COEFFS['z1']])
        self.beta0 = RAV4_2019_COEFFS['beta0']
        self.vc = RAV4_2019_COEFFS['vc']
        self.b1 = RAV4_2019_COEFFS['b1']
        self.b2 = RAV4_2019_COEFFS['b2']
        self.b3 = RAV4_2019_COEFFS['b3']
        self.fuel_type = RAV4_2019_COEFFS['fuel_type']

    def get_instantaneous_fuel_consumption(self, accel, speed, grade):
        accel_plus = np.maximum(accel, 0)
        state_variables = np.array([1,
                                    speed,
                                    speed**2,
                                    speed**3,
                                    accel,
                                    accel * speed,
                                    accel * speed**2,
                                    accel_plus ** 2,
                                    accel_plus ** 2 * speed,
                                    grade,
                                    grade * speed])
        fc = np.dot(self.state_coeffs, state_variables)
        lower_bound = (speed <= self.vc) * self.beta0
        fc = np.maximum(fc, lower_bound)
        return fc * GRAMS_PER_SEC_TO_GALS_PER_HOUR[self.fuel_type]


class NonLocalTrafficFLowHarmonizer(object):
    """A controller that gradually pushes traffic near downstream speeds."""

    def __init__(self, dt, c1=2.0, c2=0.5, th_target=2., sigma=3000.):
        """Instantiate the vehicle class."""
        self.dt = dt
        self.accel = 0.

        # =================================================================== #
        #                         tunable parameters                          #
        # =================================================================== #

        # estimation type. One of: {"uniform", "gaussian"}
        self._estimation_type = "uniform"
        # scaling term for the response to the proportional error
        self.c1 = c1
        # scaling term for the response to the differential error
        self.c2 = c2
        # target time headway, in seconds
        self.th_target = th_target
        # standard deviation for the Gaussian smoothing kernel, in meters.
        # For uniform smoothing, this acts as the smoothing window.
        self.sigma = sigma

        # =================================================================== #
        #                   lower-level control parameters                    #
        # =================================================================== #

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
        """Reset internal memory."""
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
        """Perform uniform smoothing across a window of fixed width."""
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

        # Return default speed if behind the target.
        if len(x) == 0:
            return 30.

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
        # Compute bounded acceleration.
        accel = max(
            -abs(self.max_decel), min(self.max_accel, (v_des-v_t)/self.dt))

        # Reduce noise.
        accel = self.gamma * accel + (1. - self.gamma) * self.accel

        # Make sure speed is not negative.
        accel = max(-v_t / self.dt, accel)

        return accel

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
            self.v_des = max(
                v-abs(self.max_decel)*self.dt,
                min(v+self.max_accel*self.dt, v_target))

        # Predict the average future acceleration for the leader.
        if len(self._vl) > 1:
            ix = min(len(self._vl), 50)  # 5 seconds
            _vl = list(self._vl)[-int(ix):]
            a_lead = min(0., (_vl[-1] - _vl[0]) / (self.dt*(len(_vl) - 1)))
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

        # Clip by max speed.
        self.v_des = min(self.v_des, self.v_max)

        # Compute desired acceleration.
        self.accel = self._get_accel_from_vdes(v_t=v, v_des=self.v_des)

        return self.accel


class TrafficEnv(gym.Env):

    def __init__(self, n_vehicles, av_penetration):
        self.n_vehicles = n_vehicles
        self.av_penetration = av_penetration

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(52,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def render(self, mode="human"):
        """See parent class."""
        pass

    def get_data(self):
        """Return data for saving, analysis, and plotting purposes."""
        raise NotImplementedError

    # ======================================================================= #
    #                       Visualization / evaluation                        #
    # ======================================================================= #

    def compute_metrics(self, network_type):
        assert network_type in ["i24", "bottleneck"]

        x, v, a, _, dt = self.get_data()

        # increment for AVs
        incr = int(1 if self.av_penetration == 0 else 1 / self.av_penetration)
        energy_model = PFM2019RAV4()
        mpg = []
        distance = []
        veh_count = {3500: 0, 7000: 0, 10500: 0}
        for i in range(len(x)):
            ai = a[i] = np.array(a[i])
            xi = x[i] = np.array(x[i])[:len(ai)]
            vi = v[i] = np.array(v[i])[:len(ai)]

            if network_type == "bottleneck":
                for pos in veh_count.keys():
                    if np.any(xi <= pos) and np.any(xi > pos):
                        veh_count[pos] += 1

            # Remove outside of bounds.
            if network_type == "bottleneck":
                xmin = 0
                xmax = 8000
            else:
                xmin = -float("inf")  # TODO: 0
                xmax = float("inf")

            vi = vi[(xmin <= xi) & (xi <= xmax)]
            ai = ai[(xmin <= xi) & (xi <= xmax)]
            xi = xi[(xmin <= xi) & (xi <= xmax)]

            energy = energy_model.get_instantaneous_fuel_consumption(
                speed=vi, grade=0., accel=ai)

            if len(xi) == 0:
                distance.append(0)
                mpg.append(0)
            else:
                if sum(energy) > 0:
                    dist = (xi[-1] - xi[0]) / 1609.34
                    distance.append(dist)
                    mpg.append(dist / (sum(energy) / 3600 * dt))

        n_vehicles = len(v)
        h = []
        speed = []
        th = []
        for i in range(1, n_vehicles, incr):
            # Find length of leader and ego trajectories.
            n = len(x[i-1])
            m = len(x[i])

            # Collect and store relevant data.
            coeff = get_idm_coeff(network_type)
            speed.extend(v[i])
            h_i = x[i-1][:min(n, m)] - x[i] - coeff["vlength"]
            h.extend(list(h_i))
            th_i = h_i / np.clip(v[i], a_min=1, a_max=np.inf)
            th.extend(list(th_i.flatten()[v[i].flatten() >= 1]))

        ret = {
            "tmt": np.sum(distance),
            "mpg": np.mean(mpg),
            "h_max": np.max(h),
            "h_min": np.min(h),
            "h_avg": np.mean(h),
            "th_max": np.max(th),
            "th_min": np.min(th),
            "th_avg": np.mean(th),
        }

        if network_type == "bottleneck":
            # TODO: technically the same as vehicle count since t_total is
            #  equal to one hour.
            throughput = {pos: veh_count[pos] for pos in veh_count.keys()}
            for pos in throughput.keys():
                ret[f"q_{pos}"] = throughput[pos]

        if self.av_penetration > 0:
            ret["av_mpg"] = np.mean(mpg[1::incr])

        return ret

    def gen_emission(self, network_type, emission_path):
        x, v, a, _, dt = self.get_data()

        data = {"id": [], "t": [], "x": [], "v": [], "a": []}
        for i in range(len(x)):
            xi = x[i]
            vi = v[i]
            ai = a[i]

            for t in range(len(xi)):
                data["id"].append(i)
                data["t"].append(t * dt)
                data["x"].append(xi[t])
                data["v"].append(vi[t])
                data["a"].append(ai[t] if t < len(ai) else 0.)

        os.makedirs(emission_path, exist_ok=True)

        pd.DataFrame.from_dict(data).to_csv(
            os.path.join(emission_path, "trajectory.csv"),
            float_format='%g',
            index=False)

        self.plot_statistics(network_type, emission_path)

    def plot_statistics(self, network_type, emission_path=None):
        x, v, _, v_des, dt = self.get_data()

        times = list(np.arange(len(x[0])) * dt)

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
        if emission_path is not None:
            plt.savefig(os.path.join(emission_path, "speed.png"))
        else:
            plt.show()
        plt.close()

        plt.figure(figsize=(16, 4))
        times = list(np.arange(len(x[0])) * dt)
        veh_increment = 10
        for i in range(0, len(x), veh_increment):
            plt.plot(times, np.array(x[i]) / 1000, c='k')
        plt.grid(linestyle='--')
        if network_type == "bottleneck":
            plt.ylim([0, 10.75])
        # else:  TODO
        #     plt.ylim([0, max([max(x[i] for i in range(len(x)))])])
        plt.xticks(fontsize=15)
        plt.xlabel("Time (s)", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel("Position (km)", fontsize=15)
        if emission_path is not None:
            plt.savefig(os.path.join(emission_path, "time_space.png"))
        else:
            plt.show()
        plt.close()

        if len(v_des) > 0:
            plt.figure(figsize=(16, 4))
            for i in range(len(v_des)):
                plt.plot(v_des[i], c='k')
            plt.grid(linestyle='--')
            plt.xticks(fontsize=15)
            plt.xlabel("Time (s)", fontsize=15)
            plt.yticks(fontsize=15)
            plt.ylabel("Desired speed (m/s)", fontsize=15)
            if emission_path is not None:
                plt.savefig(os.path.join(emission_path, "v_des.png"))
            else:
                plt.show()
            plt.close()
