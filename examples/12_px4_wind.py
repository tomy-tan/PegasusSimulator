#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({
    "headless": False,
    "extra_args": [
        "--/app/runLoops/main/rateLimitEnabled=true",
        "--/app/runLoops/main/rateLimitFrequency=60",
        "--/app/runLoops/main/manualModeEnabled=true",
        "--/app/player/useFixedTimeStepping=true",
    ],
})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.dynamics.wind_drag import WindDrag
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from omni.kit.viewport.utility import get_active_viewport_window
from omni.kit.widget.viewport import ViewportWidget
import omni.ui as ui

def dryden_low_altitude(w20, h_ft):
    """Turbulence settings from the MIL-F-8785C low-altitude formulas.

    w20: wind speed 20 ft above ground (m/s), also used as the steady wind speed
    h_ft: hover height above ground (ft)
    Returns (turb_sigma, turb_tau), one value each for east-west, north-south, up.
    """
    if w20 <= 0.0:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]   # no wind, no turbulence
    k = 0.177 + 0.000823 * h_ft
    sigma_w = 0.1 * w20
    sigma_uv = sigma_w / k**0.4
    L_w = h_ft * 0.3048
    L_uv = h_ft / k**1.2 * 0.3048
    return [sigma_uv, sigma_uv, sigma_w], [L_uv / w20, L_uv / w20, L_w / w20]

# Auxiliary scipy and numpy modules
import os.path
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Box Room"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": False,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        #config_multirotor.drag = WindDrag([0.50, 0.30, 0.0])

        W20 = 5.0    # m/s: light 7.72, moderate 15.43, severe 23.15
        turb_sigma, turb_tau = dryden_low_altitude(W20, h_ft=8.2)
        print("TURBCHECK", turb_sigma, turb_tau)   # temporary: remove after checking
        # Drag: Forward/Back, Left/Right, Up/Down , Wind: East North Up
        config_multirotor.drag = WindDrag(drag_coefficients=[0.50, 0.30, 0.0], steady_wind=[-W20, 0.0, 0.0],
                                          gust_start=30.0, gust_duration=3.0, gust_peak=0.0,
                                          gust_direction=[-1.0, 0.0, 0.0], turb_sigma=turb_sigma, 
                                          turb_tau=turb_tau, turb_seed=1)

        # Create camera object
        # "camera/" puts the sensor inside the RealSense model, so it moves with it
        config_multirotor.graphical_sensors = [
            MonocularCamera("camera/drone_camera", config={"frequency": 30, "depth": False, "resolution": (1280, 720),
                                                    "position": [0.148, 0.0, -0.028]})
        ]
        
        Multirotor(
            "/World/quadrotor",
            ROBOTS['Pegasus'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        
        
        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

        # Picture-in-picture: the drone camera drawn inside the main viewport.
        # Drag it by its title bar to move it.
        # "camera/" puts the sensor inside the RealSense model, so it moves with it
        pip_w, pip_h = 240, 135      # picture size in pixels (16:9)
        bar_h = 16                   # height of the title bar above the picture
        main_window = get_active_viewport_window()
        start_x = max(0, main_window.width - pip_w - 20)          # start near the bottom-right corner
        start_y = max(0, main_window.height - pip_h - bar_h - 60)
        self.pip_frame = main_window.get_frame("drone_camera_pip")
        with self.pip_frame:
            self.pip_placer = ui.Placer(draggable=True, drag_axis=ui.Axis.XY,
                                        offset_x=start_x, offset_y=start_y)
            with self.pip_placer:
                with ui.VStack(width=pip_w, height=pip_h + bar_h):
                    with ui.ZStack(height=bar_h):
                        ui.Rectangle(style={"background_color": 0xFF303030})
                        ui.Label("  Drone camera", style={"font_size": 12, "color": 0xFFDDDDDD})
                    self.pip_view = ViewportWidget(camera_path="/World/quadrotor/body/camera/drone_camera",
                                                   resolution=(pip_w, pip_h), width=pip_w, height=pip_h)

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
