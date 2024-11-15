#!/usr/bin/env python3

"""
    Just for testing some part of the code to understand what the code is doing
"""

# import board
# import digitalio
# import busio
import numpy as np
import sys

sys.path.append("../")
from exoHelper import loadParams
import exoHelper


max_torq = 53
weight = 30
rise_scalar = 40.0
torq_bound = min(max_torq, weight) #JWTODO: figure out the relation between max_torq and human body weight


""" Interpolation parameters
    Each row represents four values used in interpolation for exoskeleton torque (three rows x four columns)"""
interp_params = np.array([
    [min(torq_bound, weight * 0.64051748), 54.59, 0.58666667 * rise_scalar, 9.98],
    [min(torq_bound, weight * 0.72715882), 54.59, 0.70916667 * rise_scalar, 9.98],
    [min(torq_bound, weight * 0.80887392), 54.59, 0.85 * rise_scalar, 9.98]
    ])

# Incline paramters
incl_params = np.array([min(torq_bound, weight * 0.84887392), 54.59, 0.758 * rise_scalar, 9.98]) #0.608*rise_scalar, 9.98])
# pat fast speed opt for incline: np.array([min(torq_bound, weight*0.816), 54.59, 0.92*rise_scalar, 9.98])
# nw  = #0.82584698, 0.81831322, fw = 0.825, 0.92
static_params = np.array([min(torq_bound, weight * 0.82584698), 54.59, 0.92 * rise_scalar, 9.98]) #np.array([min(torq_bound, weight*0.816), 54.59, 0.92*rise_scalar, 9.98]) #np.array([min(torq_bound, weight*0.72715882), 54.59, 0.70916667*rise_scalar, 9.98]) #np.array([10,54.8,33.4,9.1]) #interp_params[1,:]#[41, 53.6, 24.1, 10.0]# #[46.1,54.8,33.4,9.1] -- pat DD parameters
speed_params = np.array([9.58239369128718, -24.2954102910252, 16.2100069156388]) # for converting gait duration into m/s
torq_spds = np.array([0.0, 1.07, 1.33, 1.58, 1.8])

# _,_,_,val_list = np.load('../bout_data/bout_data_val.npy')
# val_ind = 0
# print(val_list)

gen_means = np.load("../save/S01/" + 'cma/final_params.npy')
print(gen_means)
for i in range(3):
    interp_params[i,:] = loadParams(interp_params[i,:], gen_means[i,:], weight, rise_scalar)
print("Loading cma opt params:", interp_params)


offline_folder = '../save/Untethered_dualleg_2/run_2/'    # folder to save data to in offline mode
data = exoHelper.loadOffline(offline_folder)
print("offline data: \n", data)

window = 10
cnt = 0
rate = 2
torq_vec = np.zeros(window)
print("orignal torq_vec: \n", torq_vec)
ts_len = 1
for i in range(window):
    torq_vec[i] = i
print("before shifting: \n", torq_vec)
torq_vec[: 5] = torq_vec[-5:]
torq_vec[5] = 100
print("after shifting: \n", torq_vec)

# if cnt >= ts_len * rate: # filter and control once enough data to filter
#     if cnt % window == 0:
#         cnt = ts_len * rate # resets counter
#         torq_vec[: ts_len * rate] = torq_vec[-ts_len * rate :]  # shift the previous data to make room for new data
#     torq_vec[cnt] = cnt
#     cnt += 1
