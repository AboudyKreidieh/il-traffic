"""Script containing different variants of the Follower Stopper.

These controllers are used as expert representations of energy-efficient
driving, which we attempt to imitate.
"""
import numpy as np
from collections import deque

from il_traffic.experts.base import ExpertModel


class VelocityController(ExpertModel):
    """Controller for setting accelerations from desired speeds."""

    def __init__(self, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        sim_step : float
            the simulation time step
        """
        super(VelocityController, self).__init__(noise=0.)

        # maximum acceleration for autonomous vehicles, in m/s^2
        self.max_accel = 1.5
        # maximum deceleration for autonomous vehicles, in m/s^2
        self.max_decel = -3
        # simulation time step, in sec/step
        self.sim_step = sim_step

    def _get_accel_from_v_des(self, speed, v_des):
        """Compute the acceleration from the desired speed.

        Parameters
        ----------
        speed : float
            the current speed of the vehicle
        v_des : float
            the desired speed by the vehicle

        Returns
        -------
        float
            the desired acceleration
        """
        # Compute the acceleration.
        accel = (v_des - speed) / self.sim_step

        # Clip by bounds.
        accel = max(min(accel, self.max_accel), self.max_decel)

        # Apply noise.
        return self.apply_noise(accel)

    def get_action(self, speed, headway, lead_speed, **kwargs):
        """See parent class."""
        raise NotImplementedError


class PISaturation(VelocityController):
    """Control strategy that attempts to drive at the average network speed.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Attributes
    ----------
    v_history : [float]
        vehicles speeds in previous timesteps
    """

    def __init__(self, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        sim_step : float
            the simulation time step
        """
        super(PISaturation, self).__init__(sim_step=sim_step)

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.v_des = 0  # equivalent to U
        self.v_target = 0
        self.v_cmd = 0

    def _update_v_history(self, speed):
        """Update the AV's velocity history.

        Parameters
        ----------
        speed : float
            the speed of the ego vehicle
        """
        self.v_history.append(speed)

        # Maintain queue length.
        if len(self.v_history) == int(60 / self.sim_step):
            del self.v_history[0]

    def get_action(self, speed, headway, lead_speed, **kwargs):
        """See parent class."""
        dv = lead_speed - speed
        dx_s = max(2 * dv, 4)

        # Update the AV's velocity history.
        self._update_v_history(speed)

        # update desired velocity values
        self.v_des = np.mean(self.v_history)

        v_target = self.v_des + self.v_catch \
            * min(max((headway - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((headway - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_speed) \
            + (1 - beta) * self.v_cmd

        # Compute the acceleration from the desired velocity.
        return self._get_accel_from_v_des(speed=speed, v_des=self.v_cmd)


class TimeHeadwayFollowerStopper(VelocityController):
    """Vehicle control strategy that assigns a desired speed to vehicles.

    The assigned desired speeds in this object as opposed to the
    FollowerStopper class are designed to be more conservative, allowing for
    safer driving to occur.

    See: "Reachability Analysis for FollowerStopper: Safety Analysis and
    Experimental Results", in submission

    Attributes
    ----------
    v_des : float
        desired speed of the vehicles (m/s)
    v_cmd : float
        intermediary desired speed (takes into account safe behaviors)
    """

    def __init__(self, v_des, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        v_des : float
            desired speed of the vehicle (m/s)
        sim_step : float
            the simulation time step
        """
        super(TimeHeadwayFollowerStopper, self).__init__(sim_step=sim_step)

        # desired speed of the vehicle
        self.v_des = v_des

        # intermediary desired speed (takes into account safe behaviors)
        self.v_cmd = None

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5
        self.h_1 = 0.4
        self.h_2 = 1.2
        self.h_3 = 1.8

    def get_action(self, speed, headway, lead_speed, **kwargs):
        """See parent class."""
        dv_minus = min(lead_speed - speed, 0)

        dx_1 = self.dx_1_0 + 1 / (2*self.d_1) * dv_minus**2 + self.h_1 * speed
        dx_2 = self.dx_2_0 + 1 / (2*self.d_2) * dv_minus**2 + self.h_2 * speed
        dx_3 = self.dx_3_0 + 1 / (2*self.d_3) * dv_minus**2 + self.h_3 * speed
        v = min(max(lead_speed, 0), self.v_des)

        # Compute the desired velocity.
        if headway <= dx_1:
            v_cmd = 0
        elif headway <= dx_2:
            v_cmd = v * (headway - dx_1) / (dx_2 - dx_1)
        elif headway <= dx_3:
            v_cmd = v + (self.v_des - v) * (headway - dx_2) / (dx_3 - dx_2)
        else:
            v_cmd = self.v_des

        self.v_cmd = v_cmd

        # Compute the acceleration from the desired velocity.
        return self._get_accel_from_v_des(speed=speed, v_des=self.v_cmd)


class DownstreamController(ExpertModel):
    """A controller that attempts to drive at average downstream speeds.

    Attributes
    ----------
    v_target : float
        target (estimated) desired speed
    v_max : float
        maximum assigned speed
    sim_step : float
        simulation time step, in sec/step
    c1 : float
        scaling term for the response to the proportional error
    c2 : float
        scaling term for the response to the differential error
    th_target : float
        target time headway
    sigma : float
        standard deviation for the Gaussian smoothing kernel
    """

    def __init__(self, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        sim_step : float
            simulation time step, in sec/step
        """
        super(DownstreamController, self).__init__(noise=0.)

        # simulation time step

        # Follower-Stopper wrapper
        self.fs = TimeHeadwayFollowerStopper(v_des=30., sim_step=sim_step)

        # controller parameters
        self.v_target = 30.
        self.v_max = 40.
        self.sim_step = sim_step
        self._prev_th = None
        self._vl = deque(maxlen=10)

        # tunable parameters
        self.c1 = 1.0
        self.c2 = 0.0
        self.th_target = 2.
        self.sigma = 3000.

    @staticmethod
    def kernelsmooth(x0, x, z, sigma):
        """Return a kernel-smoothing average of downstream speeds."""
        densities = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -np.square(x - x0) / (2 * sigma ** 2))

        densities = densities / sum(densities)

        return sum(densities * z)

    def get_action(self, speed, headway, lead_speed, **kwargs):
        """See parent class."""
        avg_speed = kwargs.get("avg_speed", None)
        segments = kwargs.get("segments", None)
        x0 = kwargs.get("pos", None)

        # Update the buffer of lead speeds.
        self._vl.append(lead_speed)

        if avg_speed is not None:
            # Collect relevant traffic-state info.
            # - ix0: segments in front of the ego vehicle
            # - ix1: final segment, or clipped to estimate congested speeds
            ix0 = max(
                next(i for i in range(len(segments)) if segments[i]>=x0) - 2,
                0)
            try:
                ix1 = next(j for j in range(ix0 + 2, len(segments)) if
                           avg_speed[j] - np.mean(avg_speed[ix0:j]) > 15) + 1
            except StopIteration:
                ix1 = len(segments)
            segments = segments[ix0:ix1]
            avg_speed = avg_speed[ix0:ix1]

            th = headway / (speed + 1e-6)
            th = max(0., min(40., th))
            th_error = th - self.th_target

            if self._prev_th is None:
                delta_th_error = 0.
            else:
                delta_th_error = (self._prev_th - th) / self.sim_step
                self._prev_th = th

            self.v_target = \
                self.kernelsmooth(x0, segments, avg_speed, self.sigma) + \
                self.c1 * th_error + \
                self.c2 * delta_th_error

        # Update desired speed.
        max_decel = -1.0
        max_accel = 1.0
        prev_vdes = self.fs.v_des
        self.fs.v_des = max(
            prev_vdes + max_decel * self.sim_step,
            min(
                prev_vdes + max_accel * self.sim_step,
                self.v_target,
            )
        )

        # Keep within reasonable bounds.
        self.fs.v_des = max(0., min(40., self.fs.v_des))

        # Return acceleration command by follower stopper.
        return self.fs.get_action(
            speed=speed,
            lead_speed=lead_speed,
            headway=headway,
        )
