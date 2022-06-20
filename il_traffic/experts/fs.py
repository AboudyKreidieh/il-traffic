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
