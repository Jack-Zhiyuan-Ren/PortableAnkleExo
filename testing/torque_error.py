from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted

cur_dir = os.getcwd()
save_dir = cur_dir + '/../save/Torque_Untethered_Test_4/run_24/'#Untethered_dualleg_2/run_2/'#S04/run_1/'#tm6/run_1/'#Untethered_dualleg_2/run_2/'#S05/run_1/'#tm6/run_1/'#cma_treadmill2/run_5/'#Untethered_dualleg_1/run_30/'#Torque_Untethered_Test_4/run_24/'


files = natsorted(os.listdir(save_dir))
#print(files)
rate = 200

for i,f in enumerate(files):
    try:
        temp_dat = np.load(save_dir+f)
        if i == 0:
            num_sig = temp_dat.shape[0]
            file_len = temp_dat.shape[1]
            data = np.zeros((num_sig,file_len*len(files)))

        data[:,i*file_len:(i+1)*file_len] = temp_dat
    except:
        pass

print(data.shape)

meas_torq = data[0,:]
des_torq = data[1,:]
phase = data[12,:]
velmdes = data[14,:]
stance = data[7,:]
mot_temp = data[15,:]
il_comp = data[17,:]

plt.figure()
plt.plot(mot_temp, c= 'r', label='Motor temperature (C)')
np.save('mot_data',mot_temp)
plt.legend()
plt.show()

num_steps = 10000
step_cnt = 0
samples_per_step = 100
phase_pts = np.linspace(0, 100, samples_per_step)
step_data = np.zeros((num_steps, 5, samples_per_step))
step_start_ind = 0
step_end_ind = 0
torq_ind = 0

exclude_outliers = True
torque_thresh = 5.0
for i in range(1,data.shape[1]):
    if phase[i] - phase[i-1] < -80: # new step started
        step_end_ind = i-1
        des_torq_step = des_torq[step_start_ind:step_end_ind]
        meas_torq_step = meas_torq[step_start_ind:step_end_ind]
        phases_step = phase[step_start_ind:step_end_ind]
        velmdes_step = velmdes[step_start_ind:step_end_ind]
        stance_step = stance[step_start_ind:step_end_ind]
        il_step = il_comp[step_start_ind:step_end_ind]
        step_start_ind = i
        if exclude_outliers and np.mean(meas_torq_step) < torque_thresh:
            pass
        else:
            # interpolating data and saving
            step_data[step_cnt, 0, :] = np.interp(phase_pts, phases_step, des_torq_step)
            step_data[step_cnt, 1, :] = np.interp(phase_pts, phases_step, meas_torq_step)
            step_data[step_cnt, 2, :] = np.interp(phase_pts, phases_step, velmdes_step)
            step_data[step_cnt, 3, :] = np.interp(phase_pts, phases_step, stance_step)
            step_data[step_cnt, 4, :] = np.interp(phase_pts, phases_step, il_step)
            step_cnt += 1
            if torq_ind == 0 and np.mean(meas_torq_step) > torque_thresh:
                torq_ind = step_cnt
# computing RMS of individual cycles and avg
print("Steps detected:", step_cnt, "Torque steps detected:", step_cnt - torq_ind)
ignore_cycles = torq_ind # ignore this many of the first steps
avg_meas_torq = np.mean(step_data[ignore_cycles:step_cnt, 1, :], axis=0)
std_meas_torq = np.std(step_data[ignore_cycles:step_cnt, 1, :], axis=0)
avg_des_torq = np.mean(step_data[ignore_cycles:step_cnt, 0, :], axis=0)
std_des_torq = np.std(step_data[ignore_cycles:step_cnt, 0, :], axis=0)
avg_stance = np.mean(step_data[ignore_cycles:step_cnt, 3, :], axis=0)

avg_rmse = np.sqrt(np.mean((avg_meas_torq - avg_des_torq)**2))
step_rmse = np.sqrt(np.mean((step_data[:step_cnt, 0, :] - step_data[:step_cnt, 1, :])**2))
last_steps = 20
last_steps_rmse = np.sqrt(np.mean((np.mean(step_data[step_cnt-last_steps:step_cnt, 0, :],axis=0) - np.mean(step_data[step_cnt-last_steps:step_cnt, 1, :],axis=0))**2))

print("RMSE of average profiles:", avg_rmse)
print("RMSE of steps:", step_rmse)
print("RMSE of avg over last", last_steps,"steps:", last_steps_rmse)
print("MEAN torque applied:", np.mean(avg_meas_torq))

# plotting mean and std of torques
meas_c = 'r'
des_c = 'k'
# plt.figure()
# plt.plot(phase_pts, avg_meas_torq, color = meas_c, label='Measured')
# plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
# plt.plot(phase_pts, avg_stance*max(avg_des_torq), color = 'c', ls = '--', label = 'Stance')
# plt.ylabel('Torque (Nm)')
# plt.xlabel('Gait cycle (%)')
# plt.legend()
# plt.show()

# plotting all cycles

# meas_c = 'r'
# des_c = 'k'
# plt.figure()
# for i in range(torq_ind, step_cnt):
#     plt.plot(phase_pts, step_data[i, 1, :], alpha = 0.5)#color = meas_c, alpha = i*0.7/num_steps + 0.3, label='Measured')
# #plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
# plt.ylabel('Torque (Nm)')
# plt.xlabel('Gait cycle (%)')
# #plt.legend()
# plt.show()

# plt.figure()
# steps_lim = 30
# for i in range(torq_ind, steps_lim):
#     plt.plot(phase_pts, step_data[i, 4, :], alpha = 0.5)#color = meas_c, alpha = i*0.7/num_steps + 0.3, label='Measured')
# #plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# plt.plot(phase_pts, avg_des_torq*0.1, color = des_c, label='Desired')
# plt.ylabel('IL velmdes')
# plt.xlabel('Gait cycle (%)')
# #plt.legend()
# plt.show()

plt.figure()
prev_steps = 20
for i in range(step_cnt-prev_steps, step_cnt):
    plt.plot(phase_pts, step_data[i, 1, :], alpha = 0.5)#color = meas_c, alpha = i*0.7/num_steps + 0.3, label='Measured')
    plt.plot(phase_pts, step_data[i, 4, :], alpha = 0.5)
#plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
plt.plot(phase_pts, np.mean(step_data[torq_ind:torq_ind+prev_steps,1,:], axis=0), color = meas_c, ls='--', label='Measured first few steps')
plt.plot(phase_pts, np.mean(step_data[step_cnt-prev_steps:step_cnt,1,:], axis=0), color = meas_c, label='Measured last few steps')
plt.plot(phase_pts, np.mean(step_data[step_cnt-prev_steps:step_cnt,4,:], axis=0), color = 'g', label='Measured last few steps')
plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
plt.ylabel('Torque (Nm)')
plt.xlabel('Gait cycle (%)')
plt.legend()
plt.show()


# meas_c = 'r'
# des_c = 'k'
# plt.figure()
# for i in range(torq_ind, step_cnt):
#     plt.plot(phase_pts, step_data[i, 2, :], alpha = 0.3)#color = meas_c, alpha = i*0.7/num_steps + 0.3, label='Measured')
# #plt.fill_between(phase_pts, avg_meas_torq - std_meas_torq, avg_meas_torq + std_meas_torq, alpha = 0.2, color = meas_c)
# #plt.plot(phase_pts, avg_des_torq, color = des_c, label='Desired')
# plt.ylabel('Command to motor (limited to 25.6)')
# plt.xlabel('Gait cycle (%)')
# #plt.legend()
# plt.show()
