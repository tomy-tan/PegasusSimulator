import numpy as np
from pegasus.simulator.logic.dynamics.drag import Drag
from pegasus.simulator.logic.state import State
from scipy.spatial.transform import Rotation

class WindDrag(Drag):
    def __init__(self, drag_coefficients=[0.0, 0.0, 0.0], steady_wind=[0.0, 0.0, 0.0],
                 gust_start = 0.0, gust_duration = 1.0, gust_peak = 0.0, gust_direction = [0.0, 0.0, 0.0]):
        super().__init__()
    
        self._drag_coefficients = np.diag(drag_coefficients)
    
        self._drag_force = np.array([0.0, 0.0, 0.0])

        self._steady_wind = np.array(steady_wind)

        self._print_timer = 0.0

        self._gust_start = gust_start
        self._gust_duration = gust_duration
        self._gust_peak = gust_peak
        self._gust_direction = np.array(gust_direction)
        self._time = 0.0

    @property # getter
    def drag(self):

        return self._drag_force
    
    def update(self, state: State, dt: float):
        self._time += dt
        
        if self._gust_start <= self._time <= self._gust_duration + self._gust_start:
            gust_speed = ((1-np.cos((2 * np.pi) * ((self._time - self._gust_start) / self._gust_duration))) / 2) * self._gust_peak
        else:
            gust_speed = 0.0
        
        wind = self._steady_wind + gust_speed * self._gust_direction
        
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