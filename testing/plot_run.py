from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
cwd = os.getcwd()
save_dir = cwd + '/../save/S09/run_7/'#Untethered_dualleg_1/run_6/'#Stopngo/run_3/'
files = natsorted(os.listdir(save_dir))
#print(files)
rate = 200

for i,f in enumerate(files):
    try:
        temp_dat = np.load(save_dir+f, allow_pickle=True)
        if i == 0:
            num_sig = temp_dat.shape[0]
            file_len = temp_dat.shape[1]
            data = np.zeros((num_sig,file_len*len(files)))

        data[:,i*file_len:(i+1)*file_len] = temp_dat
    except:
        print("File failed to load", f)


meas_torq = data[0,:]
des_torq = data[1,:]
mot_torq = data[2,:]
phase = data[12,:]
stance = data[7,:]
il_comp = data[17,:]
times = np.linspace(0,len(meas_torq)/200., len(meas_torq)) #data[13,:]

plt.figure()
plt.plot(times, meas_torq, color = 'c', label='Measured Torque')
plt.plot(times, il_comp, color = 'k', label='IL comp')
plt.show()

# plotting torques
plt.figure()
plt.plot(times, mot_torq, color = 'y', label='Motor Torque')
plt.plot(times, des_torq, color = 'r', label='Desired Torque')
plt.plot(times, meas_torq, color = 'c', label='Measured Torque')
plt.plot(times, stance*5, color = 'k', label = 'Stance')
plt.legend()
plt.show()

# plotting insoles
plt.figure()
insole_colors = ['r','g','b','c']
insole_labels = ['I1','I2','I3','I4']
for i in range(4):
    plt.plot(times, data[3+i,:], color = insole_colors[i], label=insole_labels[i])
plt.plot(times, stance, color = 'k', label='Stance')
plt.legend()
plt.show()

# plot kinematics
ank_angle = data[9,:]
ank_vel = data[11,:]
early_stop = np.zeros(len(times))
step_cnt = np.zeros(len(times))
time_last_step = 0.0
step_len_thresh = 2.4
scheck1 = False
scheck2 = False
for i in range(1,len(times)):
    if phase[i-1] > 80.0 and phase[i] < 10.0:
        step_cnt[i] = step_cnt[i-1] + 1
        time_last_step = times[i]
        scheck1 = False
        scheck2 = False
    else:
        step_cnt[i] = step_cnt[i-1]
    if (times[i] - time_last_step) > step_len_thresh: # reset count
        step_cnt[i] = 0
    if step_cnt[i] >= 3:
        if not scheck1 and ank_vel[i] < -0.1 and phase[i] < 35.0: #ank_angle[i] > 0.0
            scheck1 = True # look to see if early ankle angle is greater than 0
        elif not scheck1 and phase[i] >= 25.0:
            early_stop[i] = 1

        #if stance[i] == 0.0 and stance[i-1] == 1.0 and ank_vel[i] < 0.4: # transition to swing
        #    early_stop[i] = 1 # check velocity

plt.figure()
torq_scalar = 0.05
plt.plot(times, des_torq*torq_scalar, color = 'k', label='Desired Torque')
plt.plot(times, meas_torq*torq_scalar, color = 'c', label='Measured Torque')
plt.plot(times, ank_angle*3, color = 'b', label='Ankle Angle')
plt.plot(times, ank_vel, color = 'r', label='Ankle Vel')
plt.plot(times, stance*0.3, color = 'k', label='Stance')
plt.plot(times, phase*0.01, color = 'g', label='phase')
plt.legend()
plt.show()

plt.figure()
plt.plot(times, ank_angle, color = 'b', label='Ankle Angle')
plt.plot(times, ank_vel, color = 'r', label='Ankle Vel')
plt.plot(times, stance*0.3, color = 'k', label='Stance')
plt.plot(times, phase*0.01, color = 'g', label='phase')
plt.plot(times, step_cnt*0.1, color = 'c', label='step cnt')
plt.plot(times, early_stop, color = 'k', label='early stop')
plt.legend()
plt.show()

# plotting angles
plt.figure()
plt.plot(times, data[8,:], color = 'r', label='Motor Angle / 5')
plt.plot(times, data[9,:], color = 'b', label='Ankle Angle')
plt.legend()
plt.show()
#
# plotting velocity
plt.figure()
plt.plot(times, data[10,:], color = 'r', label='Motor Vel')
plt.plot(times, data[11,:], color = 'b', label='Ankle Vel')
plt.legend()
plt.show()

plt.figure()
plt.plot(times, data[12,:], color = 'r', label='Motor Vel')
plt.plot(times, data[13,:], color = 'b', label='Ankle Vel')
plt.legend()
plt.show()

#
# num_steps = 1000
# step_cnt = 0
# samples_per_step = 100
# phase_pts = np.linspace(0, 100, samples_per_step)
# step_data = np.zeros((num_steps, 2, samples_per_step))
# step_start_ind = 0
# step_end_ind = 0
#
# for i in range(1,data.shape[1]):
#     if phase[i] - phase[i-1] < -80: # new step started
#         step_end_ind = i-1
#         des_torq_step = des_torq[step_start_ind:step_end_ind]
#         meas_torq_step = meas_torq[step_start_ind:step_end_ind]
#         phases_step = phase[step_start_ind:step_end_ind]
#         step_start_ind = i
#         # interpolating data and saving
#         step_data[step_cnt, 0, :] = np.interp(phase_pts, phases_step, des_torq_step)
#         step_data[step_cnt, 1, :] = np.interp(phase_pts, phases_step, meas_torq_step)
#         step_cnt += 1
#
# # computing RMS of individual cycles and avg
# print("Steps detected:", step_cnt)
# ignore_cycles = 0 # ignore this many of the first steps
# avg_meas_torq = np.mean(step_data[ignore_cycles:step_cnt, 1, :], axis=0)
# std_meas_torq = np.std(step_data[ignore_cycles:step_cnt, 1, :], axis=0)
# avg_des_torq = np.mean(step_data[ignore_cycles:step_cnt, 0, :], axis=0)
# std_des_torq = np.std(step_data[ignore_cycles:step_cnt, 0, :], axis=0)
#
# avg_rmse = np.sqrt(np.mean((avg_meas_torq - avg_des_torq)**2))
# step_rmse = np.sqrt(np.mean((step_data[:step_cnt, 0, :] - step_data[:step_cnt, 1, :])**2))
#
# print("RMSE of average profiles:", avg_rmse)
# print("RMSE of steps:", step_rmse)
#
# # plotting mean and std of torques
# meas_c = 'r'
# des_c = 'k'
# plt.figure()
# plt.plot(phase_pts, avg_meas_torq, color = meas_c, label='Measured')
# plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
# plt.ylabel('Torque (Nm)')
# plt.xlabel('Gait cycle (%)')
# plt.legend()
# plt.show()
#
# # plotting all cycles
#
# meas_c = 'r'
# des_c = 'k'
# plt.figure()
# for i in range(30, 40):
#     plt.plot(phase_pts, step_data[i, 1, :], color = meas_c, alpha = i*0.7/num_steps + 0.3, label='Measured')
# #plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
# plt.ylabel('Torque (Nm)')
# plt.xlabel('Gait cycle (%)')
# #plt.legend()
# plt.show()
