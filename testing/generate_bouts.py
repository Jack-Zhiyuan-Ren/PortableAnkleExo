import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import pickle
import os
import math

def random_bout(bout_duration, bout_cum_freq, max_time):
    pdf = np.random.uniform(0.0, 100.0)
    bout_ind = np.argwhere(pdf <= bout_cum_freq)[0][0]
    bout_dur = bout_duration[bout_ind]
    if bout_dur > max_time:
        bout_dur = max_time
    rest_dur = np.random.uniform(4.0, 7.0)
    return bout_dur, rest_dur

walk_spd_mean = 1.35
walk_spd_std = 0.17
from scipy.stats import norm
percentiles = [0.01, 0.20, 0.25, 0.33, 0.40, 0.50, 0.66, 0.75, 0.80, 0.99]
strcat = ''
for p in percentiles:
    strcat = strcat + str(np.round(walk_spd_mean + norm.ppf(p)*walk_spd_std,2)) + ', '
print(strcat)

# setup the whole opt
save_dir = os.getcwd() + '/../bout_data/' 
modelname = 'lr_model.pkl'
steps_per_cond = 23
steps_remove = 7
num_constants = 6
num_conds_per_gen = 6
num_signals = 2
num_bins = 30


# setting up bout distribution
max_time = 220 # clip results to 210 seconds if greater. 80 for val
opt_time = 0.0
total_opt_time = 45*60.0 #50*60.0, 7*60.0 fr val # total time in seconds
bout_duration = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210])
bout_freq =     np.array([20.1, 26, 14.3, 9.5, 6.5, 4.9, 3.5, 2.5, 2.1, 1.6, 1.3, 1, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.3, 3.0])
# making min bout of 30 seconds
np.random.seed(1)# 2 for val
#bout_duration = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210])
#bout_freq =     np.array([60.4, 9.5, 6.5, 4.9, 3.5, 2.5, 2.1, 1.6, 1.3, 1, 0.8, 0.7, 0.6, 0.5, 0.5, 0.4, 0.3, 0.3, 3.0])

bout_spd_full = []
bout_cum_freq = np.zeros(len(bout_freq))
strings = ['Walk at a slow speed','Walk as if you were walking a small dog','Walk as if you were walking home after a really bad day','Walk as if you were walking through a park','Walk as if you were walking in the grocery store','Walk as if you were walking from the bedroom to the kitchen','Walk as if you were walking through a field','Walk as if you were walking home with a group of friends at night','Walk as if you were walking home from the bus after class','Walk as if you were walking home from class','Walk at your typical speed','Walk at your normal speed','Walk as if you were walking a big dog','Walk as if you were walking to class','Walk as if you were going to pick up an item off of a table','Walk as if you were walking across the street','Walk as if you were walking home alone at night','Walk as if you were jay-walking','Walk as if you were walking to catch a bus', 'Wlk as if you were walking to class and you were late', 'Walk as fast as possible']
strings_speeds = [0.91, 1.06, 1.13, 1.18, 1.25, 1.27, 1.29, 1.32, 1.34, 1.36, 1.38, 1.40, 1.43, 1.46, 1.5, 1.56, 1.66, 1.77, 1.80, 1.9, 2.16]
for i, fr in enumerate(bout_freq):
    if i > 0:
        bout_cum_freq[i] = bout_cum_freq[i-1] + bout_freq[i]
    else:
        bout_cum_freq[i] = bout_freq[i]
bout_dur = [] # duration for each bout
bout_speed_list = [] # walking speeds for each bout
bout_break = [] # rests following each bout # TODO add pause to controller when skip button pressed during break
bout_command_list = [] # string of name of prompt for that speed

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx
    else:
        return array[idx], idx

num_checks = 80
speed_arr = np.linspace(walk_spd_mean-walk_spd_std*2.5, walk_spd_mean+walk_spd_std*2.5, num=num_checks)
speed_errs = np.zeros(num_checks)
gt_fit_line = scipy.stats.norm.pdf(speed_arr, walk_spd_mean, walk_spd_std)
bins = 20

#generate the conditions and desired speeds from distributions (save these) - make sure they will hopefully fill up the conditions of generations of the bins well?
while(opt_time < total_opt_time):
    dur, rest = random_bout(bout_duration, bout_cum_freq, max_time)
    bout_dur.append(dur)
    bout_break.append(rest)
    opt_time = np.sum(bout_dur)# + np.sum(bout_break)
    if dur < 0:
        spd = np.random.normal(loc=walk_spd_mean, scale=walk_spd_std)
    else:
        for i in range(num_checks):
            speed_temp = speed_arr[i]
            bout_spd_full_temp = np.copy(bout_spd_full).tolist()
            for j in range(dur):
                bout_spd_full_temp.append(speed_temp)
            #_, bin_ret, _ = plt.hist(bout_spd_full_temp, bins)# make hist
            mut, sigmat = scipy.stats.norm.fit(bout_spd_full_temp)
            best_fit_line = scipy.stats.norm.pdf(speed_arr, mut, sigmat)
            #print(best_fit_line.shape, gt_fit_line.shape, mut, sigmat)
            speed_errs[i] = np.mean(np.abs(gt_fit_line - best_fit_line))
    #         if i == num_checks-1:
    #             plt.figure()
    #             plt.plot(best_fit_line)
    #             plt.plot(gt_fit_line)
    #             plt.show()
        print(opt_time/total_opt_time)

        min_ind = np.argmin(speed_errs)
        spd = speed_arr[min_ind]
    # check a bunch of speeds to see which minimizes distance to ideal speed distribution
    #bout_speed_list.append(spd)
    nearest_spd,idx = find_nearest(strings_speeds, spd)
    bout_speed_list.append(nearest_spd)
    bout_command_list.append(strings[idx])
    for i in range(dur):
        bout_spd_full.append(nearest_spd)
    #print(dur, rest, opt_time, spd, nearest_spd, idx)

# what speed ranges contain exactly one quarter of the durations?, split into 4 windows?
sorted_spds = np.sort(bout_spd_full)
print(np.percentile(sorted_spds,0), np.percentile(sorted_spds,25), np.percentile(sorted_spds,50), np.percentile(sorted_spds,75), np.percentile(sorted_spds,100))
print(np.percentile(sorted_spds,33), np.percentile(sorted_spds,66))
print(np.percentile(sorted_spds,20), np.percentile(sorted_spds,40), np.percentile(sorted_spds,60), np.percentile(sorted_spds,80), np.percentile(sorted_spds,100))

# visualize the bouts as a histogram in terms of duration and speeds
bins = 20
plt.figure()
plt.hist(bout_dur, bins)
plt.xlabel('Duration (s)')
plt.ylabel('Number of occurences')
plt.show()

plt.figure()
plt.ylabel('Time (s)')
plt.xlabel('Desired speed (m/s)')
plt.hist(bout_spd_full, bins)
plt.show()
#plt.xlabel(
np.save(save_dir + 'bout_data_45_2.npy', [bout_dur, bout_break, bout_speed_list, bout_command_list])
