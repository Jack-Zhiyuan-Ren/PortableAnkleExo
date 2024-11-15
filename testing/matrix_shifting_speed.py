import os
import time
import numpy as np
from scipy import signal

b1,a1 = signal.butter(N=3, Wn=20, fs = 200, analog=False) # 3rd order 20hz butter
b2,a2 = signal.butter(N=3, Wn=10, fs = 200, analog=False) # 3rd order 20hz butter
z, p, k = signal.tf2zpk(b1, a1)
eps = 1e-9
r = np.max(np.abs(p))
approx_impulse_len = int(np.ceil(np.log(eps) / np.log(r)))
print('impulse', approx_impulse_len)

rate = 200 # Hz of the control and sensing loops
plot_rate = 4# Hz to update plots (faster is more comp burden)
ts_len = 4 # length of time to show in plots (s) times samples per second
window_s = 3 # number of plot windows to save to each file

data = np.zeros(rate*ts_len*window_s)
data2 = np.zeros(rate*ts_len*window_s)

start_time = time.time()
num_trials = 10000
cnt = 1
for i in range(num_trials):
    #cnt += 1
    #cnt = ts_len*rate
    data[:ts_len*rate] = data[-ts_len*rate:]
    data2[:ts_len*rate] = data2[-ts_len*rate:]

spd1 = (time.time() - start_time)/num_trials

start_time = time.time()
for i in range(num_trials):
    #cnt += 1
    #cnt = ts_len*rate
    np.roll(data, ts_len*rate)
    np.roll(data2, ts_len*rate)

spd2 = (time.time() - start_time)/num_trials

print("Avg time:", spd1, spd2, spd2/spd1)
