from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

save_dir = '/home/pi/Desktop/exo/save/filter_data/run_1/'
files = natsorted(os.listdir(save_dir))
#print(files)
file_len = 2500
data = np.zeros((12,file_len*len(files)))

for i,f in enumerate(files):
    temp_dat = np.load(save_dir+f)
    data[:,i*file_len:(i+1)*file_len] = temp_dat

rate = 250
b2,a2 = sg.butter(N=3, Wn=20, fs = rate, analog=False) # 3rd order 20hz butter
b1,a1 = sg.butter(N=2, Wn=20, fs = rate, analog=False) # 3rd order 20hz butter

raw_pos = data[8,:]
raw_torq = data[2,:]
raw_vel = (raw_pos[1:]-raw_pos[:-1])*rate
filt_pos = np.copy(raw_pos)
filt_torq = np.copy(raw_torq)
filt_torq2 = np.copy(raw_torq)
move_torq = np.copy(raw_torq)
len_data = data.shape[1]
window = 800
print(len_data)
plt_len = 8000

for i in range(2500, plt_len):
    #pass
    filt_torq_temp = sg.filtfilt(b2, a2, raw_torq[i-window:i], method='gust', irlen=100)
    filt_torq2[i] = filt_torq_temp[-1]
    filt_torq_temp = sg.filtfilt(b1, a1, raw_torq[i-window:i], method='gust', irlen=100)
    filt_torq[i] = filt_torq_temp[-1]


#filt_pos = sg.filtfilt(b2, a2, raw_pos)
#filt_torq = sg.filtfilt(b1, a1, raw_torq)
filt_vel = (filt_pos[1:]-filt_pos[:-1])*rate

# plt.figure()
# plt.plot(raw_pos[:plt_len], label='Raw')
# plt.plot(filt_pos[:plt_len], label='Filt')
# plt.ylabel('Ankle ang (rad)')
# plt.show()
# 
# 
# plt_len = 8000
# plt.figure()
# plt.plot(raw_vel[:plt_len], label='Raw')
# plt.plot(filt_vel[:plt_len], label='Filt')
# plt.ylabel('Ankle vel (rad/s)')
# plt.show()

plt_start = 5500
plt_len = plt_start+2500

move_len = 4
for i in range(move_len, plt_len):
    move_torq[i] = np.mean(raw_torq[i-move_len:i])
    
plt.figure()
plt.plot(raw_torq[plt_start:plt_len], label='Raw')
plt.plot(filt_torq[plt_start:plt_len], label='Filt')
plt.plot(filt_torq2[plt_start:plt_len], label='Filt2')
plt.plot(move_torq[plt_start:plt_len], label='move')

plt.ylabel('torq (Nm)')
plt.show()

