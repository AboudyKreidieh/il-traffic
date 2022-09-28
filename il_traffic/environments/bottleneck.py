import time
import numpy as np
import math
from pynverse import inversefunc

from il_traffic.environments.traffic import TrafficEnv
from il_traffic.environments.traffic import IDM_COEFFS
from il_traffic.environments.traffic import VEHICLE_LENGTH
from il_traffic.environments.traffic import NonLocalTrafficFLowHarmonizer


# position of the bottleneck
BN_START = 8000
BN_END = 10000
# bottleneck speed reduction coefficient
BN_COEFF = 0.8
# traffic state estimation time delay (in seconds)
T_DELAY = 60


class BottleneckEnv(TrafficEnv):

    def __init__(self, n_vehicles, av_penetration):
        super(BottleneckEnv, self).__init__(
            n_vehicles=n_vehicles,
            av_penetration=av_penetration,
        )

        # memorization for speed limit for different densities
        self._sl = {}
        # use in get_data only
        self._inverted = False
        # runtime start
        self._t0 = 0.
        # simulation step size
        self.dt = 0.1
        # human time delay, in steps
        self.delay = 0
        # time counter
        self.t = 0
        # time horizon, in steps
        self.horizon = 20000
        # equilibrium speed for a given density
        self.get_v_of_k = inversefunc(self.get_k_of_v)

        if av_penetration == 0:
            incr = 0
            n_avs = 0
        else:
            incr = int(1/av_penetration)
            n_avs = n_vehicles // incr

        # indices for automated vehicles
        self.av_indices = incr * np.arange(n_avs, dtype=int) + 1
        # expert controllers
        self.expert = [
            NonLocalTrafficFLowHarmonizer(dt=self.dt)
            for _ in range(len(self.av_indices))]
        # acceleration memory
        self.prev_accel = np.zeros(n_avs, dtype=np.float32)

        # positions where traffic state estimation data is collected
        self.x_seg = np.array([
            -2400.1983071905865,
            -1434.331821594883,
            -717.1659169049603,
            0.0,
            824.4583972637208,
            1822.6236804668458,
            2867.8068568430263,
            3912.991117494698,
            4818.99452020466,
            5723.950059403363,
            6628.90555117323,
            7533.861145531142,
            7834.118714075751,
            8413.358925562565,
            8438.816735338183,
            8997.274734087987,
            10248.728836607464,
            11153.684430774929,
            12058.63997651673,
            13175.554479957258,
            13868.551491087443,
        ], dtype=np.float32)

        # vehicle state data
        self.x = []
        self.v = []
        self.a = []
        self.all_vdes = []

        # traffic state estimation data
        # self.x_seg = []
        self.v_seg = []

    def reset(self, _add_av=True):
        print("-----------")
        print("Starting bottleneck simulation.")
        self._t0 = time.time()
        self._inverted = False

        # Initial speed of all vehicles
        v0 = IDM_COEFFS["v0"] - 5  # TODO

        # Clear data from previous runs.
        self.x = []
        self.v = []
        self.a = []
        self.all_vdes = []
        self.v_seg = v0 * np.ones(len(self.x_seg), dtype=np.float32)

        # Define initial state.
        h_eq = 1.1 * v0  # get_h_eq(v=v0, vl=v0)
        x_init = BN_START - (h_eq + VEHICLE_LENGTH) * np.arange(
            self.n_vehicles, dtype=np.float32)
        v_init = v0 * np.ones(self.n_vehicles, dtype=np.float32)
        self.x.append(x_init)
        self.v.append(v_init)

        if self.av_penetration > 0:
            # Reset expert controller and memory.
            n_avs = len(self.expert)
            for i in range(n_avs):
                self.expert[i].reset()
            self.prev_accel = np.zeros(n_avs)

            # Get initial state.
            state = self.get_state()
        else:
            state = []

        return state

    def step(self, action):  # TODO: control vehicles in a range
        """See parent class.

        Note that, if action is set to None, accelerations for the AVs will be
        provided by the non-local controller.
        """
        # current state information
        x_t = self.x[-1]
        v_t = self.v[-1]

        # Get human-driver model accelerations.
        a_t = self.get_human_accel()

        if self.av_penetration > 0:
            # Update traffic state estimation data.
            self.update_tse()

            av_indices = self.av_indices[(0 <= x_t) & (x_t <= BN_START)]
            experts = []
            for ix, expert in zip(self.av_indices, self.expert):
                if 0 <= x_t[ix] <= BN_START:
                    experts.append(expert)

            # Get expert accelerations.
            expert_accel = [expert.get_accel(
                v=v_t[ix],
                vl=v_t[ix-1],
                h=x_t[ix-1] - x_t[ix] - VEHICLE_LENGTH,
                x=x_t[ix],
                x_seg=self.x_seg,
                v_seg=self.v_seg,
            ) for ix, expert in zip(av_indices, experts)]

            # Convert to bounded desired speed differences.
            expert_vdes = np.array(
                [expert.v_des for expert in experts], dtype=np.float32)
            expert_action = [
                np.array([max(-5, min(5, expert_vdes[i] - v_t[ix]))])
                for i, ix in enumerate(av_indices)]

            # Store in memory.
            self.all_vdes.append(expert_vdes)

            # Get automated vehicle accelerations.
            if action is not None:
                a_av = self.get_av_accel(action, av_indices)
            else:
                a_av = expert_accel

            # Update accelerations in AV indices.
            for ix, a in zip(av_indices, a_av):
                a_t[ix] = a
        else:
            expert_action = None

        # Update dynamics.
        x_tp1 = x_t + v_t * self.dt + 0.5 * a_t * self.dt ** 2
        v_tp1 = v_t + a_t * self.dt

        # Append vehicle data.
        self.x.append(x_tp1)
        self.v.append(v_tp1)
        self.a.append(a_t)

        # Update time counter.
        self.t += 1
        done = self.t == self.horizon

        # Check for a collision
        h_t = x_tp1[1:] - x_tp1[:-1] - VEHICLE_LENGTH
        if any(h_t) <= 0:
            print("Collision")
            done = True

        # Return state and expert action.
        next_obs = self.get_state()
        reward = 0.
        info = {"expert_action": expert_action}

        if done:
            info.update(self.compute_metrics())
            print("\nDone. Simulation runtime: {} sec".format(
                round(time.time() - self._t0, 2)))

        return next_obs, reward, done, info

    @staticmethod
    def _get_idm_accel(v, vl, h, dt):
        n_veh = len(v)
        v0 = IDM_COEFFS["v0"]
        T = IDM_COEFFS["T"]
        a = IDM_COEFFS["a"]
        b = IDM_COEFFS["b"]
        delta = IDM_COEFFS["delta"]
        s0 = IDM_COEFFS["s0"]
        noise = IDM_COEFFS["noise"]

        s_star = s0 + np.clip(
            v * T + v * (v - vl) / (2 * math.sqrt(a * b)),
            a_min=0, a_max=np.inf)

        accel = a * (1 - (v / v0) ** delta - (s_star / h) ** 2)

        if noise > 0:
            accel += math.sqrt(dt) * np.random.randn(n_veh) * noise

        # Make sure speed is not negative.
        accel = np.clip(accel, a_min=-v/dt, a_max=np.inf)

        return accel

    def get_human_accel(self):
        """Return accelerations for vehicles if they were humans."""
        # current state information
        x_t = self.x[-1]
        v_t = self.v[-1]

        # Add acceleration of the leader (it doesn't oscillate) and the
        # vehicles following it.
        a_t = [0.] + list(self._get_idm_accel(
            v=v_t[1:],
            vl=v_t[:-1],
            h=x_t[:-1] - x_t[1:] - VEHICLE_LENGTH,
            dt=self.dt,
        ))

        # Density in veh/meter (ignoring the lead vehicle)
        n_veh = np.sum((BN_START <= x_t[1:]) & (x_t[1:] <= BN_END))
        k_bn = n_veh / (BN_END - BN_START)

        # speed limit within the bottleneck
        if k_bn > 0:
            if n_veh in self._sl:
                sl = self._sl[n_veh]
            else:
                sl = self.get_v_of_k(k_bn) * BN_COEFF
                self._sl[n_veh] = sl
        else:
            sl = IDM_COEFFS["v0"]

        # Force vehicles to slow down in the bottleneck if their
        # current speed is higher than the speed limit.
        for i in np.where(
                (x_t >= BN_START) & (x_t <= BN_END) & (v_t >= sl))[0]:
            if i != 0:
                a_t[i] = -1.

        return np.array(a_t, dtype=np.float32)

    def get_av_accel(self, action,  av_indices):
        """Convert actions to a desired acceleration."""
        n_avs = len(av_indices)

        # Make sure actions is of the same size.
        assert action.shape == (n_avs, 1)

        # failsafe parameters
        h_min = 5.
        tau = 5.
        th_min = 0.5

        # current state information
        x_t = self.x[-1][av_indices]
        v_t = self.v[-1][av_indices]
        xl_t = self.x[-1][av_indices - 1]
        vl_t = self.v[-1][av_indices - 1]
        h_t = xl_t - x_t - VEHICLE_LENGTH

        # approximation for leader acceleration
        t0 = max(0, self.t - int(tau / self.dt))
        if self.t - t0 > 0:
            a_lead = min(
                0., (self.v[-1][av_indices] -
                     self.v[t0][av_indices]) / (self.dt * (self.t - t0)))
        else:
            a_lead = np.zeros(n_avs, dtype=np.float32)

        # Get desired speed from action.
        v_des = v_t + action.flatten()

        assert v_des.shape == (n_avs,)

        # Update desired speed based on safety.
        v_des = np.clip(
            v_des,
            a_min=0.,
            a_max=(h_t
                   - h_min
                   + vl_t * tau
                   + 0.5 * a_lead * tau ** 2
                   - 0.5 * v_t * tau) / (th_min + 0.5 * tau)
        )

        # Compute bounded acceleration.
        accel = np.clip((v_des - v_t) / self.dt, a_min=-3.0, a_max=1.5)

        # Reduce noise.
        gamma = 0.5
        accel = gamma * accel + (1. - gamma) * self.prev_accel

        self.prev_accel = accel

        # Make sure speed is not negative.
        accel = np.clip(accel, a_min=-v_t/self.dt, a_max=np.inf)

        return accel

    def get_state(self, av_indices):
        """Return the agent's observation.

        This observation consists of:

        1. Ego time headway
        2. Ego speed
        3. Previous desired speed
        4. History of leader's speeds
        """
        history_length = 50
        speed_scale = 40.
        h_scale = 100.

        # current state information
        n_avs = len(av_indices)
        t = len(self.x)
        t0 = max(0, t - history_length)
        x_t = self.x[-1][av_indices]
        v_t = self.v[-1][av_indices]
        xl_t = self.x[-1][av_indices - 1]
        vl_t = [self.v[ti][av_indices - 1] for ti in range(t0, t)]
        h_t = xl_t - x_t - VEHICLE_LENGTH

        obs = []
        for i in range(n_avs):
            obs.append(np.array(
                [h_t[i] / h_scale, v_t[i] / speed_scale]
                + [0.] * (history_length - t + t0)
                + [arr[i] / speed_scale for arr in vl_t]))

        return obs

    def update_tse(self):
        if self.t > 0 and self.t % int(T_DELAY / self.dt) == 0:
            sensing_delay = 0

            # Get delayed information (reversed since vehicles are from
            # front to end).
            x = self.x[-(sensing_delay + 1)][::-1]
            v = self.v[-(sensing_delay + 1)][::-1]
            # Sort speeds by positions.
            speeds = [[] for _ in range(len(self.x_seg))]
            index = 0
            for i in range(len(x)):
                if x[i] > self.x_seg[index]:
                    index += 1
                if index == len(self.x_seg):
                    break
                speeds[index].append(v[i])

            # Update variable.
            self.v_seg = np.array(
                [np.mean(s) for s in speeds], dtype=np.float32)

    @staticmethod
    def get_k_of_v(v):
        """Get the density of the speed using IDM equilibrium."""
        T = IDM_COEFFS["T"]
        delta = IDM_COEFFS["delta"]
        s0 = IDM_COEFFS["s0"]
        v0 = IDM_COEFFS["v0"]
        s = (s0 + T * v) / (1 - (v / v0) ** delta) ** (1 / 2)
        k = 1 / (s + VEHICLE_LENGTH)
        return k

    def get_data(self):
        """Return data for saving, analysis, and plotting purposes."""
        # number of datapoints to skip when saving. For efficiency purposes.
        skip = 10

        # Convert data to appropriate format for base class.
        if not self._inverted:
            self._inverted = True
            self.x = list(np.c_[self.x[::skip]].T)
            self.v = list(np.c_[self.v[::skip]].T)
            self.a = list(np.c_[self.a[::skip]].T)
            # if len(self.all_vdes) > 0:
            #     self.all_vdes = list(np.c_[self.all_vdes[::skip]].T)

        return self.x, self.v, self.a, [], skip * self.dt
