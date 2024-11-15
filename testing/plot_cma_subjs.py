from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
from cma_helper import plot_single_mode
from scipy.stats import norm

cur_dir = os.getcwd()
plot_angs = False # set to true for individuals plots
subjs = ['S01','S02','S03','S04','S05','S06','S07','S08','S09']
save_dir = cur_dir + '/subj_plots/'
num_subjs = len(subjs)
percentiles = np.linspace(0.01,0.99,200)#[0.01, 0.05, 0.1, 0.20, 0.25, 0.33, 0.40, 0.50, 0.66, 0.75, 0.80, 0.9, 0.95, 0.99]
ppf = []
pdf=[]
strcat = ''
walk_spd_mean = 1.35
walk_spd_std = 0.17
for p in percentiles:
    ppfr = walk_spd_mean + norm.ppf(p)*walk_spd_std
    strcat = strcat + str(np.round(ppfr,2)) + ', '
    ppf.append(ppfr)
    pdf.append(norm.pdf(ppfr, walk_spd_mean, walk_spd_std))
scalar = 160/1.3
subjs_params = np.zeros((3,num_subjs,2))
subjs_sigma = np.zeros((num_subjs,3,24))
subjs_steps = []
for i, subj in enumerate(subjs):
    print(subj)
    folder = cur_dir + '/../save/'+subj+'/cma/gen_data/'
    num_files = len(os.listdir(folder))//4
    file_num = str(num_files-1)
    cur_bin = 1
    walk_speeds = []
    prev_gen_counter = np.zeros(3)
    print("Costs by generation")
    for j in range(num_files):
        try:
            cost = np.load(folder + 'cost_gen_' + str(j) + '.npy', allow_pickle = True)
        except:
            pass
        #print(cost)
        try:
            cma_gen = np.load(folder + 'cma_data_'+ str(j) + '.npy', allow_pickle = True)
        except:
            pass
        new_param, bout_counter, bin_gen_params, plot_sig_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter = cma_gen
        new_gen = gen_counter - prev_gen_counter
        prev_gen_counter = gen_counter
        cur_bin_w = np.argwhere(new_gen)
        #print("here", gen_counter, new_gen, cur_bin_w)
        try:
            walk_speed_temp = np.load(folder + 'walk_speed_'+ str(j) + '.npy', allow_pickle = True)
        except:
            pass
        walk_flat = list(walk_speed_temp[cur_bin_w,:,:].flatten())
        #print(len(walk_flat), walk_flat)
        walk_speeds = walk_speeds + walk_flat

    #print(walk_speeds)
    if plot_angs:
        plt.figure()
        plt.hist(np.array(walk_speeds)+0.1, 40)
        plt.plot(ppf, np.array(pdf)*scalar)
        plt.ylabel("Number of steps")
        plt.xlabel("Walking speed during opt (m/s)")
        plt.xlim(0.6)
        plt.show()

    subjs_steps.append(walk_speeds)

    np.save(save_dir+'ideal_speed_x.npy', ppf, allow_pickle=True)
    np.save(save_dir+'ideal_speed_y.npy', np.array(pdf), allow_pickle=True)



    cma_data = np.load(folder + '../cma_data.npy', allow_pickle = True)
    print(cma_data)
    try:
        cost_gen = np.load(folder + 'cost_gen_'+ file_num + '.npy', allow_pickle = True)
    except:
        file_num = str(num_files-2)
        cost_gen = cost
    walk_speed = np.load(folder + 'walk_speed_'+ file_num + '.npy', allow_pickle = True)
    #print(walk_speed.shape, walk_speed)
    subj_data = np.load(folder + 'subj_data_'+ file_num + '.npy', allow_pickle = True)
    print('Subj data',subj_data.shape, subj_data[0,0,:])
    new_param, bout_counter, bin_gen_params, plot_sig_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter = cma_data

    num_gens_cma = 24 # upper bound -- not predetermined # max is 50
    for j in range(plot_mean_data.shape[1]):
        if plot_mean_data[0,j,0] == 0.0:
            num_gens_cma = j-1
            break

    print("means:", plot_mean_data[:,:num_gens_cma,:])
    print("Final mean gen:", plot_mean_data[:,num_gens_cma,:])
    print(plot_sig_data.shape)
    print("Final sig gen:", plot_sig_data[:,num_gens_cma])
    subjs_params[:,i,:] = plot_mean_data[:,num_gens_cma,:]
    subjs_sigma[i,:,:num_gens_cma] = plot_sig_data[:,:num_gens_cma]
    if num_gens_cma < 24:
        subjs_sigma[i,:,num_gens_cma:] = np.expand_dims(plot_sig_data[:,num_gens_cma],axis=1)
    #print(bout_counter, gen_counter, cond_counter)

    #print("Costs:", cost_gen, cost_gen.shape)
    #print("Walk spds:", walk_speed)

    print()
    print(subj_data.shape)
    print("Height:",subj_data[cur_bin,:,1], "Weight:",subj_data[cur_bin,:,0])
    print("cmap:", subj_data[cur_bin,:,2:6])

    num_conds_per_gen = 6
    if plot_angs:
        plt.figure()
        plt.title('anle ang')
        for j in range(num_conds_per_gen):
            plt.plot(subj_data[cur_bin,j,6:36])
        plt.show()

        plt.figure()
        plt.title('anle vel')
        for j in range(num_conds_per_gen):
            plt.plot(subj_data[cur_bin,j,36:66])
        plt.show()

        bins = np.array([-np.inf, 1.24, 1.42, np.inf])#np.array([0.9, 1.24, 1.42, 1.75])#np.array([-np.inf, 1.24, 1.42, np.inf])
        spd_bins = len(bins)-1


        goal = np.ones((3,2))
        plot_rew_data = plot_sig_data
        plot_single_mode(plot_sig_data, plot_rew_data, plot_mean_data, goal, num_gens_cma, spd_bins, bins, gen_counter)
np.save(save_dir+'subjs_steps.npy', subjs_steps, allow_pickle=True)
np.save(save_dir+'subjs_params.npy', subjs_params, allow_pickle=True)
np.save(save_dir+'subjs_sigma.npy', subjs_sigma, allow_pickle=True)
