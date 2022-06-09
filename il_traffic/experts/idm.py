"""Script containing the IDM model class.

This is used to gauge human driver performance within a system.
"""
import numpy as np

from il_traffic.experts.base import ExpertModel


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

    def get_action(self, speed, headway, lead_speed, **kwargs):
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
