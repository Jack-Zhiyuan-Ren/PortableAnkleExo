# This script runs a simulated opportunistic optimization using covariance matrix adaptation.
# For simulation, a set of "optimal" parameters are provided based on random sampling.
# Running cma_test.py will perform an optimization that converges towards these optimal parameters.
# To emulate opportunistic optimization, the optimization performs update to 3 sets of parameters based on walking speed range.
# Here the walking speed is sampled from a normal distribution following real-world data.

import numpy as np
import random
import time
import numpy.matlib
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from cma_helper import init_constants, reinit_bin_upd, upd_plot_data, sample_param, f_multi, f_all, rank_params, cma_multi, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
import os

# CMA parameters
num_gens_cma = 20 # number of generations before simulation terminates
meas_noise = 0.001 # noise in CMA estimates
offset_std_in_params = 0.1 # underlying std of the "true" parameters
scalar_std_in_params = 0.1 # underlying scalar offset of "true" parameters
init_sigma_val = 0.2 # initial sigma value, controlling the covariance (similar to the range) of the first CMA generation.
speed_type = 'normal' # 'uniform'
speed_mean = 1.35 # walking speed distribution m/s for the normal distribution
speed_std = 0.15
meta_plot = []
start_time = time.time()

# specifications for optimization
seed = 3
weight = 68 # in kg
random.seed(seed)
np.random.seed(seed)
N = 2 # dimensions
f_params = np.array([[0.64051748, 0.58666667],[0.72715882,0.70916667],[0.80887392,0.85]]) # initial values for means of bins at predetermined speeds
m = 'Normal' #'Re-initialization'
bins = np.array([0.9, 1.24, 1.42, 1.75]) # defining speed ranges of the bins
bin = 0
spd_bins = len(bins)-1
sigma = init_sigma_val*np.ones(spd_bins)
λ, constants = init_constants(N, num_gens_cma)
param_bounds = np.zeros((N,2))
param_bounds[0,:] = np.array([0,1.]) # peak torque, max 1 = # kg of participant
param_bounds[1,:] = np.array([0.25,1.]) # rise time, max = 40, min = 10
rise_scalar = 40.0
new_params = np.array([0.701, 54.59, 27.8/rise_scalar, 9.98])
upd_flag = False
last_cond_ind = 0

# remove later below
f_offset = np.random.uniform(low=-offset_std_in_params, high=offset_std_in_params, size=(spd_bins, 1))
f_mult = np.random.uniform(low=1-scalar_std_in_params, high=1+scalar_std_in_params, size=1)
if speed_type == 'normal':
    cond_speeds = np.random.normal(loc=speed_mean, scale=speed_std, size=num_gens_cma*λ)
elif speed_type == 'uniform':
    speed_mean = 1.35 # walking speed distribution m/s
    speed_std = 0.15
    cond_speeds = np.random.uniform(low=speed_mean-speed_std, high=speed_mean+speed_std, size=num_gens_cma*λ)

x_mean = init_xmean(bins, N, f_params, np.array([0.8, 1.25, 1.7])) # starting param values
plot_sig_data = np.zeros((spd_bins,num_gens_cma+1))
plot_sig_data[:,0] = sigma
plot_rew_data = np.zeros((spd_bins,num_gens_cma+1))
goal = f_params*f_mult + f_offset # computing simulated "optimal" parameters
plot_rew_data[:,0] = f_all(goal, meas_noise, x_mean)
plot_mean_data = np.zeros((spd_bins,num_gens_cma+1, N))
plot_mean_data[:,0,:] = x_mean

bin_opt_vars = []
for i in range(spd_bins):
    bin_opt_vars.append(init_opt_vars(x_mean[i,:], sigma[i], N))
# simulation params
cond_counter = np.zeros((spd_bins), dtype=int)
gen_counter = np.zeros((spd_bins), dtype=int)
bin_gen_data = np.zeros((spd_bins, λ, N)) # stores the data for the conditions, or in sim just function evals
bin_gen_params = np.zeros((spd_bins, λ, N)) # stores the params for the conditions
bin_gen_params[:,0,:] = x_mean # initialize the params
constants.append(param_bounds)
constants.append(meas_noise)
constants.append(goal)

for i, spd in enumerate(cond_speeds): # when condition has finished (met length requirements)
    if i <= last_cond_ind and last_cond_ind != 0: # pass
        print('skip',i)
        sample_param(bin_opt_vars[bin], param_bounds) # so rand generated params are the same as they would have been
        continue
    bin = np.where(spd > bins)[0][-1]
    bin_gen_data[bin, cond_counter[bin], :] = bin_gen_params[bin, cond_counter[bin], :]
    cond_counter[bin] += 1
    if cond_counter[bin] % λ == 0: # generation is finished
        arindex, arx = rank_params(constants, bin, bin_gen_data[bin,:,:]) # when generation is done evaluate all those inputs to get ranking
        bin_opt_vars[bin] = cma_multi([constants, bin_opt_vars[bin]], arindex, bin_gen_params[bin,:,:]) # passing bin specific parameters to cma
        if m == 'Re-initialization':
            bin_opt_vars, upd_flag = reinit_bin_upd(bin_opt_vars, bin, len(bins)-1, bins, 2, sigma[0], upd_flag, m)
        cond_counter[bin] = 0
        gen_counter[bin] += 1
        plot_sig_data, plot_rew_data, plot_mean_data = upd_plot_data(bin_opt_vars, gen_counter, bin, plot_sig_data, plot_rew_data, plot_mean_data, len(bins)-1, constants)
    bin_gen_params[bin, cond_counter[bin], :] = sample_param(bin_opt_vars[bin], param_bounds) # sample new param from the mean/sigma for that bin
    new_params[0] = bin_gen_params[bin, cond_counter[bin], 0]*weight
    new_params[2] = bin_gen_params[bin, cond_counter[bin], 0]*rise_scalar
    print("New params:", new_params)
    cma_data = [i, bin_gen_data, bin_gen_params, plot_sig_data, plot_rew_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter]
opt_results =  bin_opt_vars

# PLOTTING
if True:
    print("Mode", m)
    plot_single_mode(plot_sig_data, plot_rew_data, plot_mean_data, goal, num_gens_cma, spd_bins, bins, gen_counter)
meta_plot.append([plot_sig_data, plot_rew_data, plot_mean_data])
print("Gen counts:", gen_counter)
print("Run time (s):", time.time() - start_time)
