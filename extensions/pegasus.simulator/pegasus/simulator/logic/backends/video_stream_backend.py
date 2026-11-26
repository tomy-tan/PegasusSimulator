"""
Streams the drone camera live as H.264 to an RTSP server (e.g. the Radxa's mediamtx).
"""
import os
import subprocess

import cv2
from pegasus.simulator.logic.backends.backend import Backend


class VideoStreamBackend(Backend):

    def __init__(self, url="rtsp://172.16.30.155:8554/sim", fps=30):
        super().__init__(config=None)    # Backend wants a config; the streamer doesn't need one

        self._url = url
        self._fps = fps
        self._proc = None            # the ffmpeg program, started when the first picture arrives (then we know its size)
        self._failed = False         # True once ffmpeg has stopped: no more frames are sent
        self._frames_sent = 0
        self._time = 0.0             # our own sim clock (s), advanced in update()
        self._first_frame_time = 0.0

    def _start_ffmpeg(self, width, height):
        cmd = [
            "ffmpeg", "-loglevel", "warning", "-nostats",
            # Input: raw BGR pictures from our stdin
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(self._fps), "-i", "-",
            # Output: H.264 with low delay and one keyframe per second, published to the RTSP server
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", str(self._fps), "-pix_fmt", "yuv420p",
            # Cap the data rate: when everything moves (take-off), quality drops a little instead of the stream overflowing
            "-b:v", "3M", "-maxrate", "3M", "-bufsize", "1M",
            "-f", "rtsp", "-rtsp_transport", "tcp", self._url,
        ]
        # Isaac Sim sets its own library paths, which can confuse the system's ffmpeg, so start it without them
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, env=env)

    def update_graphical_sensor(self, sensor_type: str, data):
        """Called by Pegasus with each new picture from a graphical sensor (30 per second for the camera)."""
        if sensor_type != "MonocularCamera" or self._failed:
            return

        frame = data["camera"].get_rgb()
        if frame is None:            # None until the first picture is rendered
            return
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Sim time on the picture, to measure the delay: compare it with the t: printed on the PC
        cv2.putText(bgr, f"t = {self._time:.2f} s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        if self._proc is None:
            height, width = bgr.shape[:2]
            self._proc = self._start_ffmpeg(width, height)
            self._frames_sent = 0
            self._first_frame_time = self._time
            print(f"t: {self._time:.2f} s  STREAM: sending to {self._url} ({width}x{height}, {self._fps} fps)")

        try:
            self._proc.stdin.write(bgr.tobytes())
            self._frames_sent += 1
        except BrokenPipeError:      # ffmpeg has stopped, e.g. it couldn't reach the server
            print(f"t: {self._time:.2f} s  STREAM: ffmpeg stopped, streaming is off (see ffmpeg's message above)")
            self._failed = True

    def update(self, dt: float):
        """Called on every physics step."""
        self._time += dt

    def stop(self):
        """Called when the simulation stops: close ffmpeg's input so it finishes and disconnects."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
            print(f"STREAM: {self._frames_sent} frames sent = {self._frames_sent / self._fps:.1f} s of video, "
                  f"{self._time - self._first_frame_time:.1f} s of sim time")

    # Required by Backend, but the streamer doesn't need them
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