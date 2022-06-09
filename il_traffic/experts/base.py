"""Script containing the base expert class.

Used to define the structure for various experts arbitrarily.
"""
import numpy as np


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

    def get_action(self, speed, headway, lead_speed, **kwargs):
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
