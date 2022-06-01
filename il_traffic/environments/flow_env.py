"""Environment for simulating a controller on the a variety of networks."""
import numpy as np
import gym
import random
import os
from copy import deepcopy
from collections import defaultdict
from csv import DictReader

from flow.envs import Env
from flow.networks import I210SubNetwork
from flow.core.params import InFlows
from flow.core.util import ensure_dir

from il_traffic.experts import FlowExpertModel
from il_traffic.experts import IntelligentDriverModel

ADDITIONAL_ENV_PARAMS = dict(
    # the controller to use
    controller_cls=None,
    # dictionary of controller params
    controller_params=dict(),
    # the interval (in meters) in which automated vehicles are controlled. If
    # set to None, the entire region is controllable.
    control_range=[500, 2500],
    # maximum allowed acceleration for the AV accelerations, in m/s^2
    max_accel=1,
    # maximum allowed deceleration for the AV accelerations, in m/s^2
    max_decel=1,
    # number of observation frames to use. Additional frames are provided from
    # previous time steps.
    obs_frames=5,
    # frames to ignore in between each delta observation
    frame_skip=5,
    # whether to use all observations from previous steps. If set to False,
    # only the past speed is used.
    full_history=False,
    # whether to include the average speed of the leader vehicle in the
    # observation
    avg_speed=False,
    # whether to save the frames of the GUI. These can be processed and coupled
    # together later to generate a video of the simulation.
    save_video=False,
)

OPEN_PARAMS = dict(
    # path to the initialized vehicle states. Each initialization also defines
    # the inflow rate and the speed limit of the downstream edge.
    warmup_path=None,
    # the AV penetration rate, defining the portion of inflow vehicles that
    # will be automated. If "inflows" is set to None, this is irrelevant.
    rl_penetration=0.05,
)

# These edges have an extra lane that RL vehicles do not traverse (since they
# do not change lanes). We as a result ignore their first lane computing per-
# lane observations.
EXTRA_LANE_EDGES = [
    "119257908#1-AddedOnRampEdge",
    "119257908#1-AddedOffRampEdge",
    ":119257908#1-AddedOnRampNode_0",
    ":119257908#1-AddedOffRampNode_0",
    "119257908#3",
]

# scaling term for the speeds
SPEED_SCALE = 1/10
# scaling term for the headways
HEADWAY_SCALE = 1/100


class ControllerEnv(Env):
    """Environment for simulating a controller on a variety of networks.

    With this class, custom vehicle controllers can be assigned to vehicles
    within a control range and after a warmup period.

    Required from env_params:

    * controller_cls: the controller to use
    * controller_params: dictionary of controller params
    * control_range: the interval (in meters) in which automated vehicles are
      controlled. If set to None, the entire region is controllable.
    * max_accel: maximum allowed acceleration for the AV accelerations, in m/s2
    * max_decel: maximum allowed deceleration for the AV accelerations, in m/s2
    * obs_frames: number of observation frames to use. Additional frames are
      provided from previous time steps.
    * save_video: whether to save the frames of the gui. These can be
      concatenated later to create videos.

    If training, the following is also required:

    * warmup_path: path to the initialized vehicle states. Each initialization
      also defines the inflow rate and the speed limit of the downstream edge.
    * rl_penetration: the AV penetration rate, defining the portion of inflow
      vehicles that will be automated. If "inflows" is set to None, this is
      irrelevant.
    """

    def __init__(self,
                 env_params,
                 sim_params,
                 network=None,
                 simulator='traci'):
        """Initialize the environment class."""
        super(ControllerEnv, self).__init__(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=simulator,
        )

        # the observations for RL vehicles in the control range for the past
        # "obs_frames" steps
        self._obs_history = defaultdict(list)
        self._leader_speed = defaultdict(list)

        # relevant data for logging per-step information
        self.mpg_vals = []
        self.mpg_times = []
        self._mpg_data = {}
        self._mean_speeds = []
        self._mean_accels = []

        # enter time of vehicles into the control range. Used to compute the
        # travel time of individual vehicles.
        self._t_enter = {}

        # experienced travel times
        self.travel_times = []

        # names of vehicles in the control range
        self._veh_ids = []

        # names of the RL vehicles in the control range
        self.rl_ids = []

        # =================================================================== #
        # Auxiliary features to save videos/screenshots.                      #
        # =================================================================== #

        if self.env_params.additional_params["save_video"]:
            # Run assertions.
            assert sim_params.render, \
                "In order to save videos, please include the flag: --render"
            assert sim_params.emission_path is not None, \
                "In order to save videos, please include the flag: " \
                "--gen_emission"

            # Create the screenshots folder.
            ensure_dir(os.path.join(sim_params.emission_path, "screenshots"))

        # =================================================================== #
        # Vehicle dynamics for the AVs in the non-RL settings.                #
        # =================================================================== #

        # dynamics controller for uncontrolled RL vehicles (mimics humans)
        human_id = "human"
        controller = self.k.vehicle.type_parameters[human_id][
            "acceleration_controller"]
        self._human_controller = controller[0](
            veh_id="av",
            car_following_params=self.k.vehicle.type_parameters[human_id][
                "car_following_params"],
            **controller[1]
        )

        # dynamics controller for controlled RL vehicles
        av_id = "av"
        self._controller_cls = env_params.additional_params["controller_cls"]
        controller_params = env_params.additional_params["controller_params"]
        self._av_controller = FlowExpertModel(
            veh_id="av",
            expert=(self._controller_cls, controller_params),
            car_following_params=self.k.vehicle.type_parameters[av_id][
                "car_following_params"],
            fail_safe=[
                "obey_speed_limit",
                "safe_velocity",
                "feasible_accel",
            ] if self._controller_cls != IntelligentDriverModel else None,
        )
        self.av_controllers_dict = {}

        # =================================================================== #
        # Features used for the reset and info_dict logging operations.       #
        # =================================================================== #

        # this is stored to be reused during the reset procedure
        self._network_cls = network.__class__
        self._network_name = deepcopy(network.orig_name)
        self._network_net_params = deepcopy(network.net_params)
        self._network_initial_config = deepcopy(network.initial_config)
        self._network_traffic_lights = deepcopy(network.traffic_lights)
        self._network_vehicles = deepcopy(network.vehicles)

        # Get the paths to all the initial state xml files
        warmup_path = env_params.additional_params.get("warmup_path")
        if warmup_path is not None:
            self._warmup_paths = [
                f for f in os.listdir(warmup_path) if f.endswith(".xml")
            ]
            self._warmup_description = defaultdict(list)
            for record in DictReader(
                    open(os.path.join(warmup_path, 'description.csv'))):
                for key, val in record.items():  # or iteritems in Python 2
                    self._warmup_description[key].append(float(val))
        else:
            self._warmup_paths = None
            self._warmup_description = None

        if isinstance(network, I210SubNetwork):
            # the name of the final edge, whose speed limit may be updated
            self._final_edge = "119257908#3"
            # maximum number of lanes to add vehicles across
            self._num_lanes = 5
        else:
            # the name of the final edge, whose speed limit may be updated
            self._final_edge = "highway_end"
            # maximum number of lanes to add vehicles across
            self._num_lanes = 1

    def _apply_rl_actions(self, rl_actions):
        """Do nothing."""
        pass

    def apply_rl_actions(self, rl_actions=None):
        """See parent class.

        This method performs the following actions:

        1. It specifies the accelerations for all controlled vehicles. Vehicles
           before the warmup period and outside the control range have
           accelerations assigned based on the "human" acceleration controller.
           Otherwise, they are assigned accelerations based on the assigned
           controller. If rl_actions is not None, actions are overridden by
           this value.
        2. It computes the observed vehicles or visualization purposes.
        """
        # Separate the RL vehicles by controlled and uncontrolled.
        controlled_rl_ids, uncontrolled_rl_ids = self.get_controlled_ids()

        for veh_id in uncontrolled_rl_ids:
            # Remove the controller if it just exited the control range.
            if veh_id in self.av_controllers_dict.keys():
                del self.av_controllers_dict[veh_id]

            # Assign accelerations to uncontrolled vehicles.
            self._human_controller.veh_id = veh_id
            acceleration = self._human_controller.get_action(self)
            self.k.vehicle.apply_acceleration(veh_id, acceleration)

        for veh_id in controlled_rl_ids:
            # Create a controller class for this vehicle if it just entered
            # the control range.
            if veh_id not in self.av_controllers_dict.keys():
                controller = deepcopy(self._av_controller)
                controller.veh_id = veh_id
                self.av_controllers_dict[veh_id] = deepcopy(controller)

            # Assign the observed vehicles.
            if self._controller_cls.__name__ in [
                    "FollowerStopper",
                    "PISaturation",
                    "TimeHeadwayFollowerStopper",
                    "IntelligentDriverModel"]:
                # vehicle in front
                self.k.vehicle.set_observed(self.k.vehicle.get_leader(veh_id))

            # Assign accelerations to controlled vehicles.
            if rl_actions is None:
                acceleration = self.av_controllers_dict[veh_id].get_action(
                    self)
                self.k.vehicle.apply_acceleration(veh_id, acceleration)

        if rl_actions is not None:
            accel = rl_actions.flatten()

            # Apply the failsafe action.
            accel = [
                self.k.vehicle.get_acc_controller(veh_id).get_action(
                    self, acceleration=accel[i])
                for i, veh_id in enumerate(self.rl_ids)
            ]

            self.k.vehicle.apply_acceleration(self.rl_ids, accel)

    def get_controlled_ids(self):
        """Return the vehicles IDS of the controlled and uncontrolled AVs.

        The actions of controlled vehicles are dictated by the assigned
        controller. The actions of the uncontrolled vehicles are assigned by
        human dynamics.

        Returns
        -------
        list of str
            controlled RL vehicles
        list of str
            uncontrolled RL vehicles
        """
        control_range = self.env_params.additional_params["control_range"]

        # In the warmup period all vehicles act as humans.
        if self.time_counter < \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            controlled_rl_ids = []
            uncontrolled_rl_ids = self.k.vehicle.get_rl_ids()

        # If no control range is specified, all vehicles are controlled.
        elif control_range is None:
            controlled_rl_ids = self.k.vehicle.get_rl_ids()
            uncontrolled_rl_ids = []

        # Vehicles in the control_range are controlled, others act as humans.
        else:
            ctrl_min, ctrl_max = control_range

            controlled_rl_ids = []
            uncontrolled_rl_ids = []
            for veh_id in self.k.vehicle.get_rl_ids():
                if ctrl_min <= self.k.vehicle.get_x_by_id(veh_id) <= ctrl_max:
                    controlled_rl_ids.append(veh_id)
                else:
                    uncontrolled_rl_ids.append(veh_id)

        return controlled_rl_ids, uncontrolled_rl_ids

    def additional_command(self):
        """See parent class."""
        control_range = self.env_params.additional_params["control_range"]

        # Get the names of the vehicles in the control range.
        if control_range is None:
            self._veh_ids = self.k.vehicle.get_ids()
        else:
            min_val = control_range[0]
            max_val = control_range[1]
            self._veh_ids = [
                veh_id for veh_id in self.k.vehicle.get_ids() if
                min_val <= self.k.vehicle.get_x_by_id(veh_id) <= max_val
            ]

        # Update the travel time for vehicles.
        self._update_travel_time(self._veh_ids)

        # Update mpg data, for visualization and logging purposes.
        self._update_mpg_vals(self._veh_ids)

    def get_state(self):
        """See parent class."""
        frames = self.env_params.additional_params["obs_frames"]
        skip = self.env_params.additional_params["frame_skip"]

        # Get the names of the controlled vehicles.
        self.rl_ids, _ = self.get_controlled_ids()

        for i, veh_id in enumerate(self.k.vehicle.get_rl_ids()):
            # Add relative observation of each vehicle.
            obs_vehicle = self._get_relative_obs(veh_id)
            self._obs_history[veh_id].append(obs_vehicle)

            # Add the speed of the lead vehicle, and maintain a maximum length
            # of 60 seconds.
            self._leader_speed[veh_id].append(obs_vehicle[-1])
            if len(self._leader_speed[veh_id]) > int(60 / self.sim_step):
                self._leader_speed[veh_id] = \
                    self._leader_speed[veh_id][-int(60 / self.sim_step):]

            # Maintain queue length.
            if len(self._obs_history[veh_id]) > frames:
                self._obs_history[veh_id] = \
                    self._obs_history[veh_id][-(frames * skip):]

        # Remove exited vehicles.
        for veh_id in list(self._obs_history.keys()):
            if veh_id not in self.k.vehicle.get_rl_ids():
                del self._obs_history[veh_id]
                del self._leader_speed[veh_id]

        obs = []
        for veh_id in self.rl_ids:
            # Initialize empty observation.
            obs_vehicle = np.array(
                [0. for _ in range(self.observation_space.shape[0])])

            if self.env_params.additional_params["full_history"]:
                # Concatenate the past n samples for a given time delta in the
                # output observations.
                obs_from_history = np.concatenate(
                    self._obs_history[veh_id][::-skip])

                # Scale values by current step information.
                if self.env_params.additional_params["obs_frames"] > 1:
                    for i in range(1, int(len(obs_from_history) / 3)):
                        obs_from_history[3 * i] = \
                            obs_from_history[3 * i] - obs_from_history[0]
                        obs_from_history[3 * i + 1] = \
                            obs_from_history[3 * i + 1] - obs_from_history[1]
                        obs_from_history[3 * i + 2] = \
                            obs_from_history[3 * i + 2] - obs_from_history[0]
                    obs_from_history[2] = \
                        obs_from_history[2] - obs_from_history[0]

                obs_vehicle[:len(obs_from_history)] = obs_from_history
            else:
                # Add the most recent observation.
                obs_vehicle[:3] = self._obs_history[veh_id][-1]

                # Add the lead speed observations for the frame skips.
                for i in range(1, frames):
                    if skip * i + 1 > len(self._obs_history[veh_id]):
                        break
                    obs_vehicle[2 + i] = \
                        self._obs_history[veh_id][-(skip * i + 1)][-1]

            # Add the average speed of the leader.
            if self.env_params.additional_params["avg_speed"]:
                obs_vehicle[-1] = np.mean(self._leader_speed[veh_id])

            # Add new observation.
            obs.append(obs_vehicle)

        return np.copy(obs)

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        vel = np.array(self.k.vehicle.get_speed(self._veh_ids))
        num_vehicles = len(self._veh_ids)

        if any(vel < -100) or kwargs["fail"] or num_vehicles == 0:
            # in case of collisions or an empty network
            reward = 0
        else:
            # Reward high system-level average speeds.
            reward_scale = 0.1
            reward = reward_scale * np.mean(vel) ** 2

        return reward

    @property
    def action_space(self):
        """Return the action space."""
        return gym.spaces.Box(
            low=-abs(self.env_params.additional_params["max_decel"]),
            high=self.env_params.additional_params["max_accel"],
            shape=(1,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        """Return the observation space."""
        if self.env_params.additional_params["full_history"]:
            # observation size times number of steps observed
            num_obs = 3 * self.env_params.additional_params["obs_frames"]
        else:
            # current observation and past speeds
            num_obs = 2 + self.env_params.additional_params["obs_frames"]

        # Add an additional element for the leader speed history.
        if self.env_params.additional_params["avg_speed"]:
            num_obs += 1

        return gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(num_obs,),
            dtype=np.float32,
        )

    def step(self, rl_actions):
        """See parent class."""
        obs, rew, done, _ = super(ControllerEnv, self).step(rl_actions)
        info = {}

        # Save the screenshot.
        if self.env_params.additional_params["save_video"]:
            self.k.kernel_api.gui.screenshot(
                "View #0",
                filename=os.path.join(
                    self.sim_params.emission_path,
                    "screenshots/{}.png".format(self.time_counter)
                )
            )

        if self.time_counter > \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            self._mean_speeds.append(np.mean(
                self.k.vehicle.get_speed(self._veh_ids, error=0)))
            self._mean_accels.append(np.mean([
                abs(self.k.vehicle.get_accel(veh_id, False, False))
                for veh_id in self._veh_ids]))
            mpg_vals = np.array(self.mpg_vals[len(self.mpg_vals) // 2:])
            tt = np.array(self.travel_times[len(self.travel_times) // 2:])

            info.update({
                "speed": np.mean(self._mean_speeds),
                "abs_accel": np.mean(self._mean_accels),
                "mpg": np.mean(mpg_vals[mpg_vals < 100]),
                "travel_time": np.mean(tt),
                "outflow": self.k.vehicle.get_outflow_rate(100),
            })

        return obs, rew, done, info

    def reset(self):
        """Perform the reset operation.

        Unique reset calls are performed for the imitation/training settings.
        """
        # Clear memory from the previous rollout.
        self.mpg_vals.clear()
        self.mpg_times.clear()
        self._mpg_data.clear()
        self._obs_history.clear()
        self._leader_speed.clear()
        self._mean_speeds = []
        self._mean_accels = []
        self._t_enter.clear()
        self.travel_times.clear()

        if self._controller_cls.__name__ in ["PISaturation"]:
            # Clear memory in the speed history buffer.
            for veh_id in self.av_controllers_dict.keys():
                self.av_controllers_dict[veh_id].v_history = []

            # Reset the timer.
            for veh_id in self.av_controllers_dict.keys():
                self.av_controllers_dict[veh_id].t = -1

        # Initialize with a random inflow/end_speed.
        if self.env_params.additional_params.get("warmup_path"):
            return self._reset_highway_i210()

        # Initialize with a define (fixed) initial condition if the
        # load_state variable is defined.
        elif self.sim_params.load_state is not None:
            _ = super(ControllerEnv, self).reset()

            self._add_automated_vehicles()

            # Add the vehicles to their respective attributes.
            self.additional_command()

            # Recompute the initial observation.
            obs = self.get_state()

            return np.copy(obs)

        return super(ControllerEnv, self).reset()

    def _reset_highway_i210(self):
        """Perform highway/I-210 specific reset operations.

        This method chooses a warmup path from the directory of available paths
        and initializes the vehicle positions/speeds as well as the network
        inflow and end speed accordingly.
        """
        end_speed = None
        params = self.env_params.additional_params
        if params["warmup_path"] is not None:
            # Make sure restart instance is set to True when resetting.
            self.sim_params.restart_instance = True

            # Choose a random available xml file.
            xml_file = random.sample(self._warmup_paths, 1)[0]
            xml_num = int(xml_file.split(".")[0])

            # Update the choice of initial conditions.
            self.sim_params.load_state = os.path.join(
                params["warmup_path"], xml_file)

            # Assign the inflow rate to match the xml number.
            lanes = 5 if self._network_cls == I210SubNetwork else 1
            inflow_rate = lanes * self._warmup_description["inflow"][xml_num]
            end_speed = self._warmup_description["end_speed"][xml_num]
            print("inflow: {}, end_speed: {}".format(inflow_rate, end_speed))

            # Create a new inflow object.
            new_inflow = InFlows()

            for inflow_i in self._network_net_params.inflows.get():
                veh_type = inflow_i["vtype"]
                edge = inflow_i["edge"]
                depart_lane = inflow_i["departLane"]
                depart_speed = inflow_i["departSpeed"]

                # Get the inflow rate of the lane/edge based on whether the
                # vehicle types are human-driven or automated.
                penetration = params["rl_penetration"]
                if veh_type == "human":
                    vehs_per_hour = inflow_rate * (1 - penetration)
                else:
                    vehs_per_hour = inflow_rate * penetration

                new_inflow.add(
                    veh_type=veh_type,
                    edge=edge,
                    vehs_per_hour=vehs_per_hour,
                    depart_lane=depart_lane,
                    depart_speed=depart_speed,
                )

            # Add the new inflows to NetParams.
            new_net_params = deepcopy(self._network_net_params)
            new_net_params.inflows = new_inflow

            # Update the network.
            self.network = self._network_cls(
                self._network_name,
                net_params=new_net_params,
                vehicles=self._network_vehicles,
                initial_config=self._network_initial_config,
                traffic_lights=self._network_traffic_lights,
            )
            self.net_params = new_net_params

            # If the expert is a FollowerStopper, update the desired speed
            # to match the free-flow speed of the new density.
            if self._controller_cls.__name__ in [
                    "FollowerStopper", "TimeHeadwayFollowerStopper"]:
                self._av_controller.expert.v_des = end_speed
                for veh_id in self.av_controllers_dict.keys():
                    self.av_controllers_dict[veh_id].expert.v_des = end_speed

        _ = super(ControllerEnv, self).reset()

        # Add automated vehicles.
        if self._warmup_paths is not None:
            self._add_automated_vehicles()

        # Update the end speed, if specified.
        if end_speed is not None:
            self.k.kernel_api.edge.setMaxSpeed(self._final_edge, end_speed)

        # Add the vehicles to their respective attributes.
        self.additional_command()

        # Recompute the initial observation.
        obs = self.get_state()

        return np.copy(obs)

    def _add_automated_vehicles(self):
        """Replace a portion of vehicles with automated vehicles."""
        penetration = self.env_params.additional_params["rl_penetration"]
        control_range = self.env_params.additional_params["control_range"]

        # Sort the initial vehicles by their positions.
        sorted_vehicles = sorted(
            self.k.vehicle.get_ids(),
            key=lambda x: self.k.vehicle.get_x_by_id(x))

        # Replace every nth vehicle with an RL vehicle.
        for lane in range(self._num_lanes):
            sorted_vehicles_lane = [
                veh for veh in sorted_vehicles if self._get_lane(veh) == lane]

            if isinstance(self.k.network.network, I210SubNetwork):
                # Choose a random starting position to allow for stochasticity.
                i = random.randint(0, int(1 / penetration) - 1)
            else:
                i = 0

            for veh_id in sorted_vehicles_lane:
                self.k.vehicle.set_vehicle_type(veh_id, "human")

                i += 1
                if i % int(1 / penetration) == 0:
                    # Don't add vehicles past the control range.
                    pos = self.k.vehicle.get_x_by_id(veh_id)
                    if pos < control_range[1]:
                        self.k.vehicle.set_vehicle_type(veh_id, "av")

    def _get_lane(self, veh_id):
        """Return a processed lane number."""
        lane = self.k.vehicle.get_lane(veh_id)
        edge = self.k.vehicle.get_edge(veh_id)
        return lane if edge not in EXTRA_LANE_EDGES else lane - 1

    def _get_gallons(self, energy_model, speed, accel, grade):
        """Calculate the instantaneous gallons consumed.

        Parameters
        ----------
        energy_model : flow.energy_models.*
            the energy model of the vehicle
        speed : float
            the speed of the vehicle
        accel : float
            the acceleration of the vehicle
        grade : int
            the road grade for the vehicle

        Returns
        -------
        float
            gallons consumed in the previous time step
        """
        if speed >= 0.0:
            gallons = energy_model.get_instantaneous_fuel_consumption(
                accel, speed, grade) * self.sim_step / 3600.0
        else:
            gallons = 0

        return gallons

    def _update_mpg_vals(self, veh_ids, dx=50):
        """Update the mpg values for the vehicles.

        This data is used for visualization purposes.

        Parameters
        ----------
        veh_ids : list of str
            the vehicles to compute the MPG values for
        dx : float
            the distance traveled after which the mpg is computed
        """
        for veh_id in veh_ids:
            # Add vehicle if missing from the mpg dict.
            if veh_id not in self._mpg_data.keys():
                self._mpg_data[veh_id] = {
                    "pos": self.k.vehicle.get_x_by_id(veh_id),
                    "gallons": 0,
                }

            speed = self.k.vehicle.get_speed(veh_id)
            accel = self.k.vehicle.get_accel(
                veh_id, noise=False, failsafe=True)
            grade = self.k.vehicle.get_road_grade(veh_id)
            pos = self.k.vehicle.get_x_by_id(veh_id)
            energy_model = self.k.vehicle.get_energy_model(veh_id)

            # Compute the gallons consumed by the next action.
            self._mpg_data[veh_id]["gallons"] += self._get_gallons(
                energy_model, speed, accel, grade)

            # Compute the distance traveled.
            distance = pos - self._mpg_data[veh_id]["pos"]

            # If distance traveled has exceeded a threshold, add the mpg to the
            # list.
            if distance > dx:
                self.mpg_times.append(
                    self.sim_step * self.time_counter)
                self.mpg_vals.append(
                    dx /
                    (1609.34 * (1e-6 + self._mpg_data[veh_id]["gallons"])))

                self._mpg_data[veh_id]["pos"] = pos
                self._mpg_data[veh_id]["gallons"] = 0

        # Remove exited vehicles.
        for veh_id in list(self._mpg_data.keys()):
            if veh_id not in self.k.vehicle.get_ids():
                del self._mpg_data[veh_id]

    def _update_travel_time(self, veh_ids):
        """Update the travel time values.

        This method does two things:

        1. If a car just entered the control range, it adds its enter time to
           the appropriate attribute.
        2. If a car just exited the control range, it computes its travel time.

        Parameters
        ----------
        veh_ids : list of str
            names of vehicles in the control range
        """
        # Compute the travel time for vehicles that exited.
        for veh_id in list(self._t_enter.keys()):
            if veh_id not in veh_ids:
                self.travel_times.append(
                    (self.time_counter-self._t_enter[veh_id]) * self.sim_step)
                del self._t_enter[veh_id]

        # Add the enter time of vehicles that entered.
        for veh_id in veh_ids:
            if veh_id not in self._t_enter:
                self._t_enter[veh_id] = self.time_counter

    def _get_relative_obs(self, veh_id):
        """Return the relative observation of a vehicle.

        The observation consists of (by index):

        1. the ego speed
        2. the headway
        3. the speed of the leader

        This also adds the leaders and followers to the vehicle class for
        visualization purposes.

        Parameters
        ----------
        veh_id : str
            the ID of the vehicle whose observation is meant to be returned

        Returns
        -------
        array_like
            the observation
        """
        obs = [None for _ in range(3)]

        # Add the speed of the ego vehicle.
        obs[0] = self.k.vehicle.get_speed(veh_id, error=10.) * SPEED_SCALE

        # Add the speed and bumper-to-bumper headway of leading vehicles.
        leader = self.k.vehicle.get_leader(veh_id)
        if leader in ["", None]:
            # in case leader is not visible
            lead_speed = 10.
            lead_head = 100.
        else:
            lead_speed = self.k.vehicle.get_speed(leader, error=10.)
            lead_head = min(
                self.k.vehicle.get_headway(veh_id, error=100.), 100.)

        obs[1] = lead_speed * SPEED_SCALE
        obs[2] = lead_head * HEADWAY_SCALE

        return obs
