"""Flow-Compatible expert model"""
from flow.controllers.base_controller import BaseController


class FlowExpertModel(BaseController):
    """Flow-compatible variant of the expert models.

    This class creates a separate sub-expert class and passes it through the
    necessary Flow controller channels.
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
            the expert class, and it's input parameters. Used to internally
            created the expert model.
        fail_safe : list < str > or str or None
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
            headway = 100.
        else:
            lead_speed = env.k.vehicle.get_speed(lead_id)

        return self.expert.get_action(
            speed=speed,
            headway=headway,
            lead_speed=lead_speed,
        )

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        raise NotImplementedError
