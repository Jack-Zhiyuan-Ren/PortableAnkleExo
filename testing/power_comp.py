import numpy as np
from chspy import CubicHermiteSpline
import os
from natsort import natsorted
import matplotlib.pyplot as plt

def torqueProfile(params, torq_bound):
    if params[0] > torq_bound:
        print("Torq error:", params[0], "Bound is:", torq_bound)
        params[0] = torq_bound
    spline = CubicHermiteSpline(n=1)
    spline.add((0, [0], [0])) # init pt
    spline.add((params[1]-params[2], [0], [0])) # start onset
    spline.add((params[1], [params[0]], [0])) # pk torque
    spline.add((params[1]+params[3], [0], [0])) # end torque
    spline.add((100, [0], [0])) # last_pt
    return spline


peak_tq_slow = np.array([4.99E+01, 5.25E+01, 5.22E+01, 5.03E+01, 5.16E+01, 4.77E+01, 5.28E+01, 5.27E+01, 5.30E+01, 5.29E+01, 4.55E+01])
peak_tq_fast = np.array([4.80E+01,5.27E+01,5.29E+01,4.84E+01,5.26E+01,5.22E+01,4.89E+01,5.25E+01,5.00E+01,5.30E+01,4.75E+01])

rise_tm_slow = np.array([2.74E+01,3.15E+01,3.06E+01,3.55E+01,2.94E+01,2.74E+01,3.07E+01,3.62E+01,3.30E+01,2.68E+01,2.84E+01])
rise_tm_fast = np.array([2.99E+01,3.60E+01,3.06E+01,3.07E+01,3.91E+01,2.99E+01,3.12E+01,4.00E+01,3.36E+01,4.00E+01,3.04E+01])

peak_tq = 0.32*peak_tq_fast + (1-0.32)*peak_tq_slow
rise_tm = 0.32*rise_tm_fast + (1-0.32)*rise_tm_slow
peak_time = 54.6
fall_time = 10.0

num_subjects = 10
num_bins = 30
gait_cycle_percent = np.linspace(0,100,num=num_bins)
ankle_ang = np.zeros((num_subjects, num_bins))
ankle_data2 = np.zeros((num_subjects, num_bins))
ankle_data = np.zeros((num_subjects, num_bins))
torque_data = np.zeros((num_subjects, num_bins))
power_data = np.zeros((num_subjects, num_bins))
tot_power_data = np.zeros(num_subjects)
tot_power_normalized_data = np.zeros(num_subjects)
cwd = os.getcwd()
save_dir = cwd + '/../save/'
save_path_2 = '/cma/subj_data.npy'
save_names = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10']# subject save folder name
subj_weight_vec = np.array([68, 65.8, 74, 53.4, 68, 76, 58, 69, 54, 84])
stride_time = np.array([1.29,1.35,1.42,1.41,1.36,1.37,1.29,1.32,1.38,1.43])
np.random.seed(1)
scaling = np.random.uniform(low=4.7, high=5.6, size=num_subjects)
for i in range(num_subjects):
    # load the ankle kinematics
    # produce the torque with some noise
    torq_prof = torqueProfile([peak_tq[i], peak_time, rise_tm[i], fall_time], 54.0)
    for j in range(num_bins):
        torque_data[i,j] = torq_prof.get_state(gait_cycle_percent[j])[0]

    # compute and plot the ankle power? see if it makes sense
    ankle_path = save_dir + save_names[i] + save_path_2
    data = np.load(ankle_path)
    temp_ankle_data = data[:,:,-30:]
    temp_ank_data = data[:,:,-60:-30]
    for j in range(3):
        angle_data = data[j,:,-30:]
        num_nonzero = 6 - np.sum(angle_data[:,15] == 0.0)
        print(num_nonzero)
        if num_nonzero < 6:
            for k in range(num_nonzero,6):
                temp_ankle_data[j,k,:] = temp_ankle_data[j,num_nonzero-1,:]
                temp_ank_data[j,k,:] = temp_ank_data[j,num_nonzero-1,:]

    time_shift = 0
    ankle_ang[i,:] = np.mean(np.mean(temp_ank_data,axis=0),axis=0)
    ankle_data[i,:] = np.mean(np.mean(temp_ankle_data,axis=0),axis=0)*scaling[i]
    ankle_data[i,:-time_shift] = ankle_data[i,time_shift:]

    ankle_data2[:,:-1] =  ankle_ang[:,1:] - ankle_ang[:,:-1]
    power_data[i,:] = ankle_data[i,:]*torque_data[i,:]


    tot_power_data[i] = np.sum(np.maximum(power_data[i,:],0.0))*(stride_time[i]/30)/stride_time[i]
    tot_power_normalized_data[i] = tot_power_data[i]/subj_weight_vec[i]
    print("Power:", tot_power_normalized_data[i])

    plt.figure()
    plt.plot(gait_cycle_percent,torque_data[i,:], label='Torque (Nm)')
    plt.plot(gait_cycle_percent,ankle_data[i,:]*50, label='Ankle angular velocity (rad/s) * 50')
    plt.plot(gait_cycle_percent,power_data[i,:], label='Ankle Power')
    plt.legend()
    plt.show()

print(np.mean(tot_power_normalized_data))
# save the ankle and torque data
np.savetxt(cwd+'/power/'+'torque_data.csv', torque_data, delimiter = ',')
np.savetxt(cwd+'/power/'+'power_data.csv', power_data, delimiter = ',')
np.savetxt(cwd+'/power/'+'ankle_data.csv', ankle_data, delimiter = ',')
np.savetxt(cwd+'/power/'+'tot_power_data.csv', tot_power_data, delimiter = ',')
np.savetxt(cwd+'/power/'+'tot_power_normalized_data.csv', tot_power_normalized_data, delimiter = ',')
np.savetxt(cwd+'/power/'+'subj_weight_vec.csv', subj_weight_vec, delimiter = ',')
np.savetxt(cwd+'/power/'+'gait_cycle_percent.csv', gait_cycle_percent, delimiter = ',')
np.savetxt(cwd+'/power/'+'stride_time.csv', stride_time, delimiter = ',')
