"""Collection of human-driver and expert models."""
import numpy as np

from flow.controllers.base_controller import BaseController


# =========================================================================== #
#                              Base expert class                              #
# =========================================================================== #

class ExpertModel(object):
    """Base expert class.

    Attributes
    ----------
    noise : float
        standard deviation of noise to assign to the accelerations
    """

    def __init__(self, noise):
        """Instantiate the expert model.

        Parameters
        ----------
        noise : float
            standard deviation of noise to assign to the accelerations
        """
        self.noise = noise

    def get_action(self, speed, headway, lead_speed):
        """Return the desired acceleration by the model.

        Parameters
        ----------
        speed : float
            the speed of the ego vehicle
        headway : float
            the bumper-to-bumper gap with the lead vehicle
        lead_speed : float
            the speed of the lead vehicle

        Returns
        -------
        float
            the desired acceleration
        """
        raise NotImplementedError

    def apply_noise(self, accel):
        """Apply gaussian noise to the desired accelerations.

        Parameters
        ----------
        accel : float
            the current acceleration

        Returns
        -------
        float
            the noisy acceleration
        """
        return accel + np.random.normal(loc=0, scale=self.noise)


# =========================================================================== #
#                             Human-driver model                              #
# =========================================================================== #

class IntelligentDriverModel(ExpertModel):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Attributes
    ----------
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    """

    def __init__(self,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 noise=0):
        """Instantiate the controller.

        Parameters
        ----------
        v0 : float
            desirable velocity, in m/s (default: 30)
        T : float
            safe time headway, in s (default: 1)
        a : float
            max acceleration, in m/s2 (default: 1)
        b : float
            comfortable deceleration, in m/s2 (default: 1.5)
        delta : float
            acceleration exponent (default: 4)
        s0 : float
            linear jam distance, in m (default: 2)
        noise : float
            standard deviation of noise to assign to the accelerations
        """
        super(IntelligentDriverModel, self).__init__(noise)

        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_action(self, speed, headway, lead_speed):
        """See parent class."""
        # in order to deal with ZeroDivisionError
        if abs(headway) < 1e-3:
            headway = 1e-3

        if lead_speed is None:  # no car ahead
            s_star = 0
        else:
            s_star = self.s0 + max(
                0,
                speed * self.T + speed * (speed - lead_speed) /
                (2 * np.sqrt(self.a * self.b)))

        accel = self.a * (
            1 - (speed / self.v0) ** self.delta - (s_star / headway) ** 2)

        return self.apply_noise(accel)


# =========================================================================== #
#                          Autonomous vehicle models                          #
# =========================================================================== #

class VelocityController(ExpertModel):
    """Controller for setting accelerations from desired speeds."""

    def __init__(self, max_accel, max_decel, noise, sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        max_accel : float
            maximum acceleration by the vehicle (in m/s2)
        max_decel : float
            maximum decelerations by the vehicle (in m/s2)
        noise : float
            standard deviation of noise to assign to the accelerations
        sim_step : float
            the simulation time step
        """
        super(VelocityController, self).__init__(noise)

        # maximum acceleration for autonomous vehicles, in m/s^2
        self.max_accel = max_accel

        # maximum deceleration for autonomous vehicles, in m/s^2
        self.max_decel = -abs(max_decel)

        # simulation time step, in sec/step
        self.sim_step = sim_step

    def _get_accel_from_v_des(self, speed, v_des):
        """Compute the acceleration from the desired speed.

        Parameters
        ----------
        speed : float
            the current speed of the vehicle
        v_des : float
            the desired (goal) speed by the vehicle

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

    def get_action(self, speed, headway, lead_speed):
        """See parent class."""
        raise NotImplementedError


class FollowerStopper(VelocityController):
    """Vehicle control strategy that assigns a desired speed to vehicles.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Attributes
    ----------
    v_des : float
        desired speed of the vehicles (m/s)
    v_cmd : float
        intermediary desired speed (takes into account safe behaviors)
    """

    def __init__(self,
                 v_des,
                 max_accel,
                 max_decel,
                 noise,
                 sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        v_des : float
            desired speed of the vehicle (m/s)
        max_accel : float
            maximum acceleration by the vehicle (in m/s2)
        max_decel : float
            maximum decelerations by the vehicle (in m/s2)
        noise : float
            standard deviation of noise to assign to the accelerations
        sim_step : float
            the simulation time step
        """
        super(FollowerStopper, self).__init__(
            max_accel=max_accel,
            max_decel=max_decel,
            noise=noise,
            sim_step=sim_step,
        )

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

    def get_vdes(self, speed, headway, lead_speed):
        """TODO."""
        dv_minus = min(lead_speed - speed, 0)
        dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus ** 2

        if headway < dx_3:
            return self.v_des
        else:
            return self.v_des + 0.001 * (headway - dx_3) ** 2

    def get_action(self, speed, headway, lead_speed):
        """See parent class."""
        dv_minus = min(lead_speed - speed, 0)

        dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus ** 2
        dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus ** 2
        dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus ** 2
        v = min(max(lead_speed, 0), self.v_des)

        # Compute the desired velocity.
        if headway <= dx_1:
            v_cmd = 0
        elif headway <= dx_2:
            v_cmd = v * (headway - dx_1) / (dx_2 - dx_1)
        elif headway <= dx_3:
            v_cmd = v + (self.v_des - v) * (headway - dx_2) / (dx_3 - dx_2)
        else:
            v_cmd = self.v_des + 0.001 * (headway - dx_3) ** 2

        self.v_cmd = v_cmd

        # Compute the acceleration from the desired velocity.
        return self._get_accel_from_v_des(speed=speed, v_des=self.v_cmd)


class PISaturation(VelocityController):
    """Control strategy that attempts to drive at the average network speed.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Attributes
    ----------
    meta_period : int
        assignment period for the desired (goal) speed
    v_history : [float]
        vehicles speeds in previous timesteps
    t : int
        number of actions performed (used for meta period)
    """

    def __init__(self, max_accel, max_decel, noise, sim_step, meta_period):
        """Instantiate the controller.

        Parameters
        ----------
        max_accel : float
            maximum acceleration by the vehicle (in m/s2)
        max_decel : float
            maximum decelerations by the vehicle (in m/s2)
        noise : float
            standard deviation of noise to assign to the accelerations
        sim_step : float
            the simulation time step
        meta_period : int
            desired speed assignment period
        """
        super(PISaturation, self).__init__(
            max_accel=max_accel,
            max_decel=max_decel,
            noise=noise,
            sim_step=sim_step,
        )

        # assignment period for the desired (goal) speed
        self.meta_period = meta_period

        # history used to determine AV desired velocity
        self.v_history = []

        # number of actions performed (used for meta period)
        self.t = -1

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

    def get_action(self, speed, headway, lead_speed):
        """See parent class."""
        self.t += 1

        dv = lead_speed - speed
        dx_s = max(2 * dv, 4)

        # Update the AV's velocity history.
        self._update_v_history(speed)

        # update desired velocity values
        if self.t % self.meta_period == 0:
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

    See: TODO

    Attributes
    ----------
    v_des : float
        desired speed of the vehicles (m/s)
    v_cmd : float
        intermediary desired speed (takes into account safe behaviors)
    """

    def __init__(self,
                 v_des,
                 max_accel,
                 max_decel,
                 noise,
                 sim_step):
        """Instantiate the controller.

        Parameters
        ----------
        v_des : float
            desired speed of the vehicle (m/s)
        max_accel : float
            maximum acceleration by the vehicle (in m/s2)
        max_decel : float
            maximum decelerations by the vehicle (in m/s2)
        noise : float
            standard deviation of noise to assign to the accelerations
        sim_step : float
            the simulation time step
        """
        super(TimeHeadwayFollowerStopper, self).__init__(
            max_accel=max_accel,
            max_decel=max_decel,
            noise=noise,
            sim_step=sim_step,
        )

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

    def set_vdes(self, speed, headway, lead_speed, v_des):
        """TODO."""
        dv_minus = min(lead_speed - speed, 0)
        dx_3 = self.dx_3_0 + 1 / (2*self.d_3) * dv_minus**2 + self.h_3 * speed

        if headway < dx_3:
            self.v_des = v_des
        else:
            self.v_des = max(v_des - 0.001 * (headway - dx_3) ** 2, 0)

    def get_vdes(self, speed, headway, lead_speed):
        """TODO."""
        dv_minus = min(lead_speed - speed, 0)
        dx_3 = self.dx_3_0 + 1 / (2*self.d_3) * dv_minus**2 + self.h_3 * speed

        if headway < dx_3:
            return self.v_des
        else:
            return self.v_des + 0.001 * (headway - dx_3) ** 2

    def get_action(self, speed, headway, lead_speed):
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
            v_cmd = self.v_des + 0.001 * (headway - dx_3) ** 2

        self.v_cmd = v_cmd

        # Compute the acceleration from the desired velocity.
        return self._get_accel_from_v_des(speed=speed, v_des=self.v_cmd)


# =========================================================================== #
#                         Flow-Compatible Expert Model                        #
# =========================================================================== #

class FlowExpertModel(BaseController):
    """Flow-compatible variant of the expert models.

    This class creates a separate sub-expert class and passes it through the
    necessary Flow controller channels.

    Attributes
    ----------
    expert : ExpertModel
        the expert model to use
    """

    def __init__(self,
                 veh_id,
                 expert,
                 fail_safe=None,
                 car_following_params=None):
        """Instantiate the model.

        Parameters
        ----------
        veh_id : str
            Vehicle ID for SUMO identification
        expert : (type [ Expert Model ], dict)
            the expert class and it's input parameters. Used to internally
            created the expert model.
        fail_safe : list of str or str or None
            type of flow-imposed failsafe the vehicle should posses, defaults
            to no failsafe (None)
        car_following_params : flow.core.params.SumoCarFollowingParams
            object defining sumo-specific car-following parameters
        """
        super(FlowExpertModel, self).__init__(
            veh_id,
            car_following_params,
            delay=0.0,
            fail_safe=fail_safe,
            noise=0,  # noise is applied on the expert model side
            display_warnings=False,
        )

        # Create the expert model.
        self.expert = expert[0](**expert[1])

    def get_accel(self, env, **kwargs):
        """See parent class."""
        # Collect some state information.
        speed = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        headway = env.k.vehicle.get_headway(self.veh_id)

        if lead_id is None or lead_id == '':  # no car ahead
            # Set some default terms.
            lead_speed = speed
            headway = 100
        else:
            lead_speed = env.k.vehicle.get_speed(lead_id)

        return self.expert.get_action(
            speed=speed,
            headway=headway,
            lead_speed=lead_speed,
        )

    def get_vdes(self, env):
        """TODO."""
        # Collect some state information.
        speed = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        headway = env.k.vehicle.get_headway(self.veh_id)

        if lead_id is None or lead_id == '':  # no car ahead
            # Set some default terms.
            lead_speed = speed
            headway = 100
        else:
            lead_speed = env.k.vehicle.get_speed(lead_id)

        return self.expert.get_vdes(
            speed=speed,
            headway=headway,
            lead_speed=lead_speed,
        )

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        raise NotImplementedError
