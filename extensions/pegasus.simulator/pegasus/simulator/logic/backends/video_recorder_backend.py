"""
Saves the drone camera's pictures to an mp4 file while the simulation runs.
"""
import os
import time

import cv2
from pegasus.simulator.logic.backends.backend import Backend


class VideoRecorderBackend(Backend):

    def __init__(self, folder="~/sim-tools/recordings", fps=30):
        super().__init__(config=None)    # Backend wants a config; the recorder doesn't need one

        self._folder = os.path.expanduser(folder)    # OpenCV doesn't understand ~
        os.makedirs(self._folder, exist_ok=True)     # VideoWriter can't create folders
        self._fps = fps
        self._writer = None          # made when the first picture arrives (then we know its size)
        self._path = None
        self._frames_written = 0
        self._time = 0.0             # our own sim clock (s), advanced in update()
        self._first_frame_time = 0.0

    def update_graphical_sensor(self, sensor_type: str, data):
        """Called by Pegasus with each new picture from a graphical sensor (30 per second for the camera)."""
        if sensor_type != "MonocularCamera":
            return

        frame = data["camera"].get_rgb()
        if frame is None:            # None until the first picture is rendered
            return
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._writer is None:
            height, width = bgr.shape[:2]
            # A new file for each run, named after the start time, so recordings don't overwrite each other
            # mkv still plays if a run ends abruptly (an mp4 needs release() to write its index at the end)
            self._path = os.path.join(self._folder, time.strftime("flight_%Y%m%d_%H%M%S.mkv"))
            self._writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (width, height))
            if not self._writer.isOpened():
                print(f"VIDEO: could not open {self._path}")
            self._frames_written = 0
            self._first_frame_time = self._time
            print(f"t: {self._time:.2f} s  VIDEO: recording to {self._path} ({width}x{height}, {self._fps} fps)")

        self._writer.write(bgr)
        self._frames_written += 1

    def update(self, dt: float):
        """Called on every physics step."""
        self._time += dt

    def stop(self):
        """Called when the simulation stops. Without release() the mp4 won't play."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"VIDEO: {self._frames_written} frames = {self._frames_written / self._fps:.1f} s of video, "
                  f"{self._time - self._first_frame_time:.1f} s of sim time, saved to {self._path}")

    # Required by Backend, but the recorder doesn't need them
    def update_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state):
        pass

    def input_reference(self):
        return []                    # motor commands only come from the first backend

    def start(self):
        pass

    def reset(self):
        pass