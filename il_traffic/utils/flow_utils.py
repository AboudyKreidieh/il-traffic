"""Utility methods for instantiating the environment and controller."""
import os

import flow.config as flow_config
from flow.controllers import ContinuousRouter
from flow.controllers import RLController
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import InFlows
from flow.networks import HighwayNetwork
from flow.networks import I210SubNetwork
from flow.networks.highway import ADDITIONAL_NET_PARAMS as HIGHWAY_NET_PARAMS
from flow.utils.registry import make_create_env
from flow.controllers import IDMController
from flow.energy_models.poly_fit_autonomie import PFMMidsizeSedan
from flow.energy_models.poly_fit_autonomie import PFM2019RAV4

import il_traffic.config as config
from il_traffic import ControllerEnv
from il_traffic import IntelligentDriverModel
from il_traffic import FollowerStopper
from il_traffic import PISaturation
from il_traffic import TimeHeadwayFollowerStopper


def get_emission_path(controller_type, network_type, network_params):
    """Assign a path for the emission data.

    Parameters
    ----------
    controller_type : int
        the controller used for the AVs in the simulation
    network_type : str
        the type of network employed
    network_params : dict
        dictionary of network-specific parameters

    Returns
    -------
    str
        the path to the emission directory.
    """
    # Specify the emission path, based on the name of the network/controller.
    if controller_type == 0:
        controller_name = "IDM"
    elif controller_type == 1:
        controller_name = "FollowerStopper"
    elif controller_type == 2:
        controller_name = "PISaturation"
    elif controller_type == 3:
        controller_name = "TimeHeadwayFollowerStopper"
    else:
        controller_name = None  # not applicable

    inflow = network_params["inflow"]
    end_speed = network_params["end_speed"]
    emission_path = "./expert_data/{}/{}/{}-{}".format(
        network_type, controller_name, int(inflow), int(end_speed))

    return emission_path


def get_network_params(inflow, end_speed, penetration_rate):
    """Return the network parameters from the argument parser.

    Parameters
    ----------
    inflow : float
        the inflow rate of vehicles (human and automated)
    end_speed : float
        the maximum speed at the downstream boundary edge
    penetration_rate : float
        penetration rate of the AVs. 0.10 corresponds to 10%

    Returns
    -------
    dict
        network-specific parameters based on choice of network
    """
    return {
        "inflow": inflow,
        "end_speed": end_speed,
        "penetration_rate": penetration_rate,
    }


def get_expert_params(network_type,
                      controller_type,
                      network_params,
                      noise,
                      verbose):
    """Get the controller parameters and control range for a given expert.

    Parameters
    ----------
    network_type : str
        the type of network to simulate. Must be one of {"ring", "highway",
        "i210"}.
    network_params : dict
        dictionary of network-specific parameters
    controller_type : int
        the type of controller, must be one of:
          (0) -- Intelligent Driver Model
          (1) -- FollowerStopper
          (2) -- PISaturation
          (3) -- TimeHeadwayFollowerStopper
    noise : float
        the standard deviation of noise assigned to accelerations by the AVs.
    verbose : bool
        whether to print relevant logging data

    Returns
    -------
    type [ il_traffic.core.experts.ExpertModel ]
        the expert model class. Used to create the model itself.
    dict
        dictionary of controller input parameters
    [float, float] or None
        the control range. If set to None, all regions are controllable.
    """
    # Specify the control range based on the choice of network.
    if network_type == "highway":
        control_range = [500, 2300]
    elif network_type == "i210":
        control_range = [573.08, 2363.27]
    else:
        control_range = None  # unknown network type

    # Add the noise term to all controllers.
    controller_params = {"noise": noise}

    # Initialize controller params with some data in case you are using the
    # velocity controllers.
    if controller_type > 0:
        controller_params.update({
            "meta_period": 10,
            "max_accel": 1,
            "max_decel": 1,
            "sim_step": 0.4,
        })

    if controller_type == 0:
        # Use the IDM.
        controller_cls = IntelligentDriverModel

        # Add controller-specific parameters.
        controller_params.update({"a": 1.3, "b": 2.0})

        if verbose:
            print("Running Intelligent Driver Model.")

    elif controller_type == 1:
        # Use the FollowerStopper.
        controller_cls = FollowerStopper

        # This term is not needed.
        del controller_params["meta_period"]

        # Compute the FollowerStopper desired speed.
        v_des = network_params["end_speed"]

        # Add controller-specific parameters.
        controller_params.update({"v_des": v_des})

        if verbose:
            print("Running FollowerStopper with v_des: {}.".format(v_des))

    elif controller_type == 2:
        # Use the PISaturation controller.
        controller_cls = PISaturation

        if verbose:
            print("Running PISaturation.")

    elif controller_type == 3:
        # Use the FollowerStopper.
        controller_cls = TimeHeadwayFollowerStopper

        # This term is not needed.
        del controller_params["meta_period"]

        # Compute the FollowerStopper desired speed.
        v_des = network_params["end_speed"]

        # Add controller-specific parameters.
        controller_params.update({"v_des": v_des})

        if verbose:
            print("Running TimeHeadwayFollowerStopper with v_des: {}.".format(
                v_des))

    else:
        raise ValueError("Unknown controller type: {}".format(controller_type))

    return controller_cls, controller_params, control_range


def get_base_env_params(network_type,
                        network_params,
                        controller_type,
                        save_video,
                        noise,
                        verbose):
    """Return the environment-specific parameters for the non-RL case.

    These are used to specify the controller used by the AVs.

    Parameters
    ----------
    network_type : str
        the type of network to simulate. Must be one of {"highway", "i210"}.
    network_params : dict
        dictionary of network-specific parameters
    controller_type : int
        the type of controller, must be one of:
          (0) -- Intelligent Driver Model
          (1) -- FollowerStopper
          (2) -- PISaturation
          (3) -- TimeHeadwayFollowerStopper
    save_video : bool
        whether to save the frames of the GUI. These can be processed and
        coupled together later to generate a video of the simulation.
    noise : float
        the standard deviation of noise assigned to accelerations by the AVs.
    verbose : bool
        whether to print relevant logging data

    Returns
    -------
    dict
        environment-specific parameters
    """
    # Get the controller parameters and control range.
    controller_cls, controller_params, control_range = get_expert_params(
        network_type=network_type,
        controller_type=controller_type,
        network_params=network_params,
        noise=noise,
        verbose=verbose,
    )

    return dict(
        # the controller to use
        controller_cls=controller_cls,
        # dictionary of controller params
        controller_params=controller_params,
        # the interval (in meters) in which automated vehicles are controlled.
        # If set to None, the entire region is controllable.
        control_range=control_range,
        # maximum allowed acceleration for the AV accelerations, in m/s^2
        max_accel=1,
        # maximum allowed deceleration for the AV accelerations, in m/s^2
        max_decel=1,
        # number of observation frames to use. Additional frames are provided
        # from previous time steps.
        obs_frames=5,
        # frames to ignore in between each delta observation
        frame_skip=5,
        # whether to use all observations from previous steps. If set to False,
        # only the past speed is used.
        full_history=False,
        # whether to include the average speed of the leader vehicle in the
        # observation
        avg_speed=False,
        # whether to save the frames of the GUI. These can be processed and
        # coupled together later to generate a video of the simulation.
        save_video=save_video,
    )


def get_rl_env_params(env_name):
    """Assign environment parameters based on the choice of environment.

    Parameters
    ----------
    env_name : str
        the name of the environment

    Returns
    -------
    dict
        addition environment parameters to provide to the environment
    """
    # Run assertion.
    assert env_name in [
        "highway", "i210"], "Unknown environment: {}".format(env_name)

    # Compute the additional dictionary parameters.
    env_params = {
        "train_vdes": True,  # TODO
        "warmup_path": os.path.join(
            config.PROJECT_PATH, "warmup/{}".format(env_name)),
        'rl_penetration': 0.05,
    }

    return env_params


def get_flow_params(network_type,
                    network_params,
                    environment_params,
                    render,
                    emission_path,
                    use_warmup=False,
                    training=False):
    """Return the flow-specific parameters when running ControllerEnv.

    Parameters
    ----------
    network_type : str
        the type of network to simulate. Must be one of {"highway", "i210"}.
    network_params : dict
        dictionary of network-specific parameters. The following parameters
        must be specified:

        * inflow (float): the inflow rate of vehicles (human and automated)
        * end_speed (float): the maximum speed at the downstream boundary edge
        * penetration_rate (float): penetration rate of the AVs. 0.10
          corresponds to 10%
    environment_params : dict
        dictionary of environment-specific parameters
    render : bool
        whether to render the environment during simulation time
    emission_path : str
        the path to the folder containing the emission file
    use_warmup : bool
        specifies whether to use a warmup file when initializing a simulation.
        This only works for a subset of cases, and will return an error message
        otherwise.
    training : bool
        whether the flow_params dict is being used for training purposes. In
        this case, load_states are added and the horizon is shrunk.

    Returns
    -------
    dict
        flow-specific parameters
    """
    # Initialize an empty inflows object.
    inflows = InFlows()

    if network_type == "highway":
        # Use the highway network class.
        network_cls = HighwayNetwork

        # Initialize additional network params.
        additional_net_params = HIGHWAY_NET_PARAMS.copy()

        # Specify a few custom parameters, including the ones provided as input
        # variables.
        additional_net_params.update({
            # length of the highway
            "length": 2500,
            # number of lanes
            "lanes": 1,
            # speed limit for all edges
            "speed_limit": 30,
            # number of edges to divide the highway into
            "num_edges": 2,
            # whether to include a ghost edge. This edge is provided a
            # different speed limit.
            "use_ghost_edge": True,
            # speed limit for the ghost edge
            "ghost_speed_limit": network_params["end_speed"],
            # length of the downstream ghost edge with the reduced speed limit
            "boundary_cell_length": 300,
        })

        inflow_rate = network_params["inflow"]
        penetration_rate = network_params["penetration_rate"]

        # Add the inflows based on the network type.
        inflows.add(
            veh_type="human",
            edge="highway_0",
            vehs_per_hour=int(inflow_rate * (1 - penetration_rate)),
            depart_lane="free",
            depart_speed=24.1,
            name="idm_inflow")
        inflows.add(
            veh_type="av",
            edge="highway_0",
            vehs_per_hour=int(inflow_rate * penetration_rate),
            depart_lane="free",
            depart_speed=24.1,
            name="av_inflow")

    elif network_type == "i210":
        # Use the I-210 network class.
        network_cls = I210SubNetwork

        # Initialize additional network params
        additional_net_params = {
            "on_ramp": False,
            "ghost_edge": True,
        }

        inflow_rate = network_params["inflow"]
        penetration_rate = network_params["penetration_rate"]

        # Add the inflows based on the network type.
        inflows.add(
            veh_type="human",
            edge="ghost0",
            vehs_per_hour=inflow_rate * 5 * (1 - penetration_rate),
            depart_lane="best",
            depart_speed=25.5,
            name="human_inflow")
        inflows.add(
            veh_type="av",
            edge="ghost0",
            vehs_per_hour=inflow_rate * 5 * penetration_rate,
            depart_lane="best",
            depart_speed=25.5,
            name="av_inflow")

    else:
        raise ValueError("Unknown network type: {}".format(network_type))

    if use_warmup:
        end_speed = network_params["end_speed"]
        penetration_rate = network_params["penetration_rate"]

        if end_speed not in [5, 6, 7, 8, 9, 10]:
            raise ValueError(
                "Only end speeds of [5, 6, 7, 8, 9, 10] are valid when trying "
                "to simulate from a warmup state.")
        elif inflow_rate not in [
                1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300]:
            raise ValueError(
                "Only inflow rates  of [1900, 1950, 2000, 2050, 2100, 2150, "
                "2200, 2250, 2300] are valid when trying to simulate from a "
                "warmup state.")

        xml_num = 2 * (int((inflow_rate - 1900) / 50) + 9 * (end_speed - 5))
        load_state = os.path.join(
            config.PROJECT_PATH,
            "warmup/{}/{}.xml".format(network_type, int(xml_num)))

        # Add the penetration rate to the environment params. This is used
        # during the reset procedure in the environment.
        environment_params["rl_penetration"] = penetration_rate

    else:
        # no load state is used in this case
        load_state = None

    # Add the humans and AVs.
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "a": 1.3,
            "b": 2.0,
            "noise": 0.3,
            "display_warnings": False,
            "fail_safe": [
                'obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            speed_mode=12,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode="sumo_default",
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=0,
        energy_model=PFMMidsizeSedan,
    )
    vehicles.add(
        "av",
        acceleration_controller=(RLController, {
            "fail_safe": [
                'obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            speed_mode=12,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=0,  # no lane changes
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=0,
        energy_model=PFM2019RAV4,
    )

    flow_params = dict(
        # name of the experiment
        exp_tag=network_type,

        # name of the flow environment the experiment is running on
        env_name=ControllerEnv,

        # name of the network class the experiment is running on
        network=network_cls,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            render=render,
            use_ballistic=True,
            overtake_right=True,
            sim_step=0.4,
            emission_path=emission_path,
            load_state=load_state,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=0 if (training or load_state is not None) else 9000,
            horizon=1500 if training else 3000,
            additional_params=environment_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflows,
            template=os.path.join(
                flow_config.PROJECT_PATH,
                "examples/exp_configs/templates/sumo/i210_with_ghost_cell_with"
                "_downstream.xml") if network_type == "i210" else None,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon init/reset
        # (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    return flow_params


def create_env(network_type,
               network_params,
               environment_params,
               render,
               emission_path=None,
               use_warmup=False,
               training=False,
               warmup_steps=None):
    """Create and return a Flow environment based on the given parameters.

    Parameters
    ----------
    network_type : str
        the type of network employed
    network_params : dict
        dictionary of network-specific parameters
    environment_params : dict
        dictionary of environment-specific parameters
    render : bool
        whether to render the environment
    emission_path : str or None
        the path to the folder containing the emission file
    use_warmup : bool
        specifies whether to use a warmup file when initializing a simulation
    training : bool
        whether the environment is being used for training purposes. In this
        case, load_states are added and the horizon is shrunk.
    warmup_steps : int or None
        number of initial warmup steps, optional. Used for testing purposes.

    Returns
    -------
    flow.Env
        the Flow environment
    """
    # Get flow parameters.
    flow_params = get_flow_params(
        network_type=network_type,
        network_params=network_params,
        environment_params=environment_params,
        render=render,
        emission_path=emission_path,
        use_warmup=use_warmup,
        training=training,
    )

    if warmup_steps is not None:
        flow_params["env"].warmup_steps = warmup_steps

    # Create the environment.
    env_creator, _ = make_create_env(flow_params)
    env = env_creator()

    return env
