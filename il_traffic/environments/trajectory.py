import time
import bisect
import numpy as np
import pandas as pd
import os
import json

import il_traffic.config as config
from il_traffic.environments.traffic import TrafficEnv
from il_traffic.environments.traffic import VEHICLE_LENGTH
from il_traffic.environments.traffic import get_idm_accel
from il_traffic.environments.traffic import get_h_eq
from il_traffic.environments.traffic import NonLocalTrafficFLowHarmonizer


def get_trajectory_from_path(fp, dt):
    df = pd.read_csv(fp)
    v_leader = np.array(df.Velocity / 3.6)

    a_leader = list((v_leader[1:] - v_leader[:-1]) / dt)
    v_leader = list(v_leader)
    x_leader = [0.]
    for t in range(1, len(v_leader)):
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


class TrajectoryEnv(TrafficEnv):

    def __init__(self, n_vehicles, av_penetration):
        super(TrajectoryEnv, self).__init__(
            n_vehicles=n_vehicles,
            av_penetration=av_penetration,
        )

        # runtime start
        self._t0 = 0.
        # simulation step size
        self.dt = 0.1
        # human time delay, in steps
        self.delay = 0
        # platoon index
        self._platoon_ix = 0
        # a rollout counter, to choose with trajectory to run
        self._rollout = 0
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
            # '2021-04-22-12-47-13_2T3MWRFVXLW056972_masterArray_0_7050',
        ]

        # expert controller
        self.expert = NonLocalTrafficFLowHarmonizer(dt=self.dt)

        # acceleration memory
        self.prev_accel = 0.

        # vehicle state data
        self.x = []
        self.v = []
        self.a = []
        self.all_vdes = []

        # traffic state estimation data
        self.x_seg = []
        self.v_seg = []
        self.t_seg = []

    def reset(self, _add_av=True):
        # Choose a trajectory.
        fp = self.fp[self._rollout % len(self.fp)]
        directory = os.path.join(
            config.PROJECT_PATH, 'warmup/i24/{}'.format(fp))

        print("-----------")
        print("Starting: {}".format(fp))
        self._t0 = time.time()

        # Increment rollout counter.
        self._rollout += 1

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

        if _add_av and self.av_penetration > 0:
            # Place an automated vehicle.
            v_eq = v_leader[0]
            h_eq = get_h_eq(v=v_eq, vl=v_leader[0])

            self.a.append([])
            self.v.append([v_eq])
            self.x.append([x_leader[0] - h_eq - VEHICLE_LENGTH])
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
        """See parent class.

        Note that, if action is set to None, accelerations for the AVs will be
        provided by the non-local controller.
        """
        done = False

        if self.av_penetration > 0:
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
            expert_accel = self.expert.get_accel(
                v=v_av[t],
                vl=v_leader[t],
                h=x_leader[t] - x_av[t] - VEHICLE_LENGTH,
                x=x_av[t],
                x_seg=self.x_seg,
                v_seg=self.v_seg[ix_t],
            )
            expert_vdes = self.expert.v_des
            expert_action = max(-5, min(5, expert_vdes - v_av[t]))
            self.all_vdes[-1].append(expert_vdes)

            # Get AV's acceleration.
            accel = expert_accel if action is None else self.get_av_accel(
                action)

            # Update dynamics.
            x_tp1 = x_av[-1] + v_av[-1] * self.dt + 0.5 * accel * self.dt ** 2
            v_tp1 = v_av[-1] + accel * self.dt

            self.x[-1].append(x_tp1)
            self.v[-1].append(v_tp1)
            self.a[-1].append(accel)
        else:
            expert_action = None

        # Check if automated vehicle is done computing its actions.
        if self.av_penetration == 0 or len(self.v[-1]) == len(self.v[0]):
            self._platoon_ix += 1

            veh_to_add = self.n_vehicles if self.av_penetration == 0 else min(
                self.n_vehicles -
                self._platoon_ix, int(1 / self.av_penetration) - 1)

            for _ in range(veh_to_add):
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

            # Check if enough vehicles were collected.
            done = self._platoon_ix >= self.n_vehicles

            if (not done) and (self.av_penetration > 0):
                # Initialize next automated vehicle.
                x_leader = self.x[-1]
                v_leader = self.v[-1]
                v_eq = v_leader[0]
                h_eq = 3.0 * v_eq

                self.a.append([])
                self.v.append([v_eq])
                self.x.append([x_leader[0] - h_eq - VEHICLE_LENGTH])
                self.all_vdes.append([])

                # Reset expert controller and memory.
                self.expert.reset()
                self.prev_accel = 0.

        # Check for a collision.
        h_t = self.x[-2][len(self.x[-1]) - 1] - self.x[-1][-1] - VEHICLE_LENGTH
        if np.any(h_t <= 0):
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
        info = {"expert_action": [np.array([expert_action])]}

        if done:
            info.update(self.compute_metrics())
            print("\nDone. Simulation runtime: {} sec".format(
                round(time.time() - self._t0, 2)))

        return next_obs, reward, done, info

    def get_av_accel(self, action):
        """Convert actions to a desired acceleration."""
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
        if t - t0 > 0:
            a_lead = min(
                0., (self.v[-2][t] - self.v[-2][t0]) / (self.dt * (t - t0)))
        else:
            a_lead = 0.

        # Get desired speed from action.
        v_des = min(40., v_t + action[0][0])

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

        # Make sure speed is not negative.
        accel = max(-v_t / self.dt, accel)

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
        h_scale = 100.

        t = len(x_av)
        t0 = max(0, t - history_length)

        x_t = x_av[-1]
        v_t = v_av[-1]
        h_t = x_leader[t-1] - x_t - VEHICLE_LENGTH

        obs = np.array(
            [h_t / h_scale, v_t / speed_scale]
            + [0.] * (history_length - t + t0)
            + [val / speed_scale for val in v_leader[t0:t]])

        return [obs]

    def get_data(self):
        """Return data for saving, analysis, and plotting purposes."""
        return self.x, self.v, self.a, self.all_vdes, self.dt
