import numpy as np
import omni.usd
from pegasus.simulator.logic.dynamics.drag import Drag
from pegasus.simulator.logic.state import State
from scipy.spatial.transform import Rotation

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

class WindDrag(Drag):
    def __init__(self, drag_coefficients=[0.0, 0.0, 0.0], steady_wind=[0.0, 0.0, 0.0],
                 gust_start = 0.0, gust_duration = 1.0, gust_peak = 0.0, gust_direction = [0.0, 0.0, 0.0],
                 turb_sigma=[0.0, 0.0, 0.0], turb_tau=[1.0, 1.0, 1.0], turb_seed=0, wind_prim_path=None,
                 show_arrow=False):
        
        super().__init__()
    
        self._drag_coefficients = np.diag(drag_coefficients)
    
        self._drag_force = np.array([0.0, 0.0, 0.0])

        self._steady_wind = np.array(steady_wind)

        self._print_timer = 0.0

        # Gust
        self._gust_start = gust_start
        self._gust_duration = gust_duration
        self._gust_peak = gust_peak
        self._gust_direction = np.array(gust_direction)
        self._time = 0.0

        self._turb_sigma = np.array(turb_sigma)
        self._turb_tau = np.array(turb_tau)
        self._rng = np.random.default_rng(turb_seed)
        self._turb = np.zeros(3)

        # Optional Wind object in the stage whose attributes override the settings above
        self._wind_prim = None
        if wind_prim_path is not None:
            self._wind_prim = omni.usd.get_context().get_stage().GetPrimAtPath(wind_prim_path)
        self._read_timer = 0.0
        self._last_settings = None

        # Optional wind arrow drawn above the drone (screen only, no physics).
        # The Wind object's "wind:show_arrow" checkbox switches it on and off while running.
        self._show_arrow = show_arrow
        self._draw = None
        if show_arrow or self._wind_prim is not None:
            from isaacsim.util.debug_draw import _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
        self._arrow_timer = 0.0

        self._wind_on = True
        
    @property # getter
    def drag(self):

        return self._drag_force

    def _read_wind_settings(self):
        """Read the Wind object's attributes and apply any changes."""
        speed = self._wind_prim.GetAttribute("wind:speed").Get()
        from_deg = self._wind_prim.GetAttribute("wind:from_deg").Get()
        turbulence = self._wind_prim.GetAttribute("wind:turbulence").Get()

        settings = (speed, from_deg, turbulence)
        if settings != self._last_settings:
            self._last_settings = settings
            # Compass angles go clockwise from north, so east uses sin and north uses cos
            towards = np.radians(from_deg + 180.0)
            self._steady_wind = np.array([speed * np.sin(towards), speed * np.cos(towards), 0.0])
            if speed > 0.0:
                self._gust_direction = self._steady_wind / speed
            if turbulence:
                sigma, tau = dryden_low_altitude(speed, h_ft=8.2)
            else:
                sigma, tau = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
            self._turb_sigma, self._turb_tau = np.array(sigma), np.array(tau)
            print(f"t: {self._time:.2f} s  WIND SETTINGS: {speed} m/s from {from_deg} deg, turbulence {turbulence}")

        if self._wind_prim.GetAttribute("wind:gust_now").Get():
            self._gust_start = self._time
            self._wind_prim.GetAttribute("wind:gust_now").Set(False)
            print(f"t: {self._time:.2f} s  GUST started")

        show_attr = self._wind_prim.GetAttribute("wind:show_arrow")
        if show_attr.IsValid():
            show = show_attr.Get()
            if show != self._show_arrow:
                self._show_arrow = show
                if not show:
                    self._draw.clear_lines()    # remove the last arrow from the screen

        enable_attr = self._wind_prim.GetAttribute("wind:enabled")
        if enable_attr.IsValid():
            enabled = enable_attr.Get()
            if enabled != self._wind_on:
                self._wind_on = enabled
                print(f"t: {self._time:.2f} s  WIND {'ON' if enabled else 'OFF'}")
        
    def update(self, state: State, dt: float):
        self._read_timer += dt
        if self._wind_prim is not None and self._read_timer >= 0.1:
            self._read_timer = 0.0
            self._read_wind_settings()

        self._time += dt
        
        if self._gust_start <= self._time <= self._gust_duration + self._gust_start:
            gust_speed = ((1-np.cos((2 * np.pi) * ((self._time - self._gust_start) / self._gust_duration))) / 2) * self._gust_peak
        else:
            gust_speed = 0.0

         # Turbulence: filtered noise, one value per world axis (east, north, up)
        a = np.exp(-dt / self._turb_tau)
        b = self._turb_sigma * np.sqrt(1 - a**2)
        self._turb = a * self._turb + b * self._rng.standard_normal(3)

        wind = self._steady_wind + gust_speed * self._gust_direction + self._turb

        if not self._wind_on:
            wind = np.zeros(3)    # wind switched off: drag comes from the drone's own motion only
        
        # Update wind arrow
        self._arrow_timer += dt
        if self._draw is not None and self._show_arrow and self._arrow_timer >= 0.1:
            self._arrow_timer = 0.0
            self._draw_wind_arrow(state.position, wind)

        # Drone's velocity relative to the ground, in the drone's axes (forward, left, up)
        body_vel = state.linear_body_velocity 

        # Wind's velocity relative to the ground, turned from world axes into the drone's axes
        wind_body = Rotation.from_quat(state.attitude).inv().apply(wind) 

        # Drone's velocity relative to the air, in the drone's axes: what drag depends on
        air_velocity = body_vel - wind_body

        self._drag_force = -np.dot(self._drag_coefficients, air_velocity)

        self._print_timer += dt
        if self._print_timer >= 0.25:
            # True attitude from the simulator (FLU axes: positive pitch = nose down)
            yaw, pitch, roll = Rotation.from_quat(state.attitude).as_euler("ZYX", degrees=True)
            print(f"t: {self._time:.2f} s  wind: {np.round(wind, 2)}  air velocity: {np.round(air_velocity, 2)}  drag: {np.round(self._drag_force, 2)} N  true pitch: {pitch:.2f} deg")
            self._print_timer = 0.0

        
            
        return self._drag_force

    def _draw_wind_arrow(self, position, wind):
        """Draw the current wind as an arrow 1 m above the drone, 0.2 m long per m/s of wind."""
        self._draw.clear_lines()
        speed = np.linalg.norm(wind)
        if speed < 0.1:
            return
        start = np.array(position) + np.array([0.0, 0.0, 1.0])
        end = start + 0.2 * wind                      # points where the air is going
        d = wind / speed                              # direction of length 1
        side = np.array([-d[1], d[0], 0.0])           # sideways, level with the ground
        head = 0.3 * 0.2 * speed                      # arrowhead is 30% of the arrow
        tip1 = end - head * d + 0.5 * head * side
        tip2 = end - head * d - 0.5 * head * side
        color = (0.0, 0.8, 1.0, 1.0)                  # light blue (red, green, blue, opacity)
        self._draw.draw_lines([tuple(start), tuple(end), tuple(end)],
                              [tuple(end), tuple(tip1), tuple(tip2)],
                              [color] * 3, [4.0] * 3)