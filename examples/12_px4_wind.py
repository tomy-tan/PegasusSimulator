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
from pegasus.simulator.logic.dynamics.wind_drag import WindDrag, dryden_low_altitude
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from omni.kit.viewport.utility import get_active_viewport_window
from omni.kit.widget.viewport import ViewportWidget
import omni.ui as ui
from pxr import Sdf
import cv2
from pegasus.simulator.logic.backends.video_recorder_backend import VideoRecorderBackend

# Auxiliary scipy and numpy modules
import os.path
from scipy.spatial.transform import Rotation

RECORD_VIDEO = False

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


        # Wind settings object, editable in the Property panel while the simulation runs
        wind_prim = self.world.stage.DefinePrim("/World/Wind", "Xform") # "Xform" is a container with a position but no shape
        wind_prim.CreateAttribute("wind:enabled", Sdf.ValueTypeNames.Bool).Set(True)   # untick to switch all wind off
        wind_prim.CreateAttribute("wind:speed", Sdf.ValueTypeNames.Float).Set(5.0)        # m/s
        wind_prim.CreateAttribute("wind:from_deg", Sdf.ValueTypeNames.Float).Set(90.0)    # compass direction the wind comes from
        wind_prim.CreateAttribute("wind:turbulence", Sdf.ValueTypeNames.Bool).Set(False)  # tick to turn turbulence on
        wind_prim.CreateAttribute("wind:gust_now", Sdf.ValueTypeNames.Bool).Set(False)    # tick to fire a gust
        wind_prim.CreateAttribute("wind:show_arrow", Sdf.ValueTypeNames.Bool).Set(False)   # tick to show the wind arrow
        

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
        # PX4 must come first: the first backend drives the motors
        backends = [PX4MavlinkBackend(mavlink_config)]

        # Optional video recorder for the drone camera (None = not recording)
        self.recorder = None
        if RECORD_VIDEO:
            self.recorder = VideoRecorderBackend()
            backends.append(self.recorder)

        config_multirotor.backends = backends

        #config_multirotor.drag = WindDrag([0.50, 0.30, 0.0])

        W20 = 5.0    # m/s: light 7.72, moderate 15.43, severe 23.15
        turb_sigma, turb_tau = dryden_low_altitude(W20, h_ft=8.2)
        print("TURBCHECK", turb_sigma, turb_tau)   # temporary: remove after checking
        # Drag: Forward/Back, Left/Right, Up/Down , Wind: East North Up
        config_multirotor.drag = WindDrag(drag_coefficients=[0.50, 0.30, 0.0], steady_wind=[-W20, 0.0, 0.0],
                                          gust_start=30.0, gust_duration=3.0, gust_peak=0.0,
                                          gust_direction=[-1.0, 0.0, 0.0], turb_sigma=turb_sigma, 
                                          turb_tau=turb_tau, turb_seed=1, wind_prim_path="/World/Wind", show_arrow=True)

        # Drone camera. "camera/" puts the sensor inside the RealSense model, so it moves with it.
        # Kept on self so run() can read its frames.
        self.drone_camera = MonocularCamera("camera/drone_camera", config={"frequency": 30, "depth": False, "resolution": (1280, 720),
                                                                           "position": [0.148, 0.0, -0.028]})
        config_multirotor.graphical_sensors = [self.drone_camera]
        
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

        self.got_frame = False

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

            if not self.got_frame:
                cam = self.drone_camera.state.get("camera")
                if cam is not None:
                    frame = cam.get_rgb() # ask cam for an RGB frame
                    if frame is not None:    # None until the first picture is rendered
                        print(f"frame shape: {frame.shape}, dtype: {frame.dtype}")
                        final_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(os.path.expanduser("~/sim-tools/recordings/frame.png"), final_frame):
                            print("Failed to save frame.")
                        self.got_frame = True

        
        # Cleanup and stop
        if self.recorder is not None:
            self.recorder.stop()    # finish the video; closing the window may shut down before Pegasus calls stop()
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
