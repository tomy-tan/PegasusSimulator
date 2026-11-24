import numpy as np

dt = 0.004           # 250 steps per second, like the simulation
duration = 3600.0    # one hour of simulated turbulence
sigma = 0.5        # m/s (horizontal)
tau = 0.5          # s

a = np.exp(-dt / tau)
b = sigma * np.sqrt(1 - a**2)

rng = np.random.default_rng(seed=1)   # fixed seed: the same "random" numbers every run
n = int(duration / dt)
noise = rng.standard_normal(n)
turb = np.zeros(n)
for k in range(1, n):
    turb[k] = a * turb[k-1] + b * noise[k]  # your filter line, using noise[k] as the random number

lag = int(tau / dt)
print("mean:", turb.mean(), "target: about 0")
print("std:", turb.std(), "target:", sigma)
print("similarity after tau:", np.corrcoef(turb[:-lag], turb[lag:])[0, 1], "target: about 0.37")