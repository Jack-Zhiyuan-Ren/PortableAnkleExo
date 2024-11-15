import time
import numpy as np
import os
from cmaHelper import grabMeans, init_constants, reinit_bin_upd, upd_plot_data, sample_param, cma_multi, cma_update, init_opt_vars, init_xmean, constrain_params, plot_single_mode
from cmaHelper import getParams, sendParams, paramGen, label_pairwise_mat, compute_ordering_from_pairs, classifyModel#, play_audio

def cma(save_dir, gen_thresh, height, weight, c, e, torq_bound, rise_scalar):
    # cma params
    print("Torq bound is", torq_bound, "Nm/kg")
    audio = True # set to True to play audio to earbuds
    cma_delay = 0.08 # delay this amount in seconds at end of loop
    seed = 1
    np.random.seed(seed)
    motor_ready = False
    N = 2
    param_scalar = 0.95
    #[0.64, 0.727, 0.81]
    f_params = np.array([[0.64051748*param_scalar, 0.58666667],[0.72715882*param_scalar,0.70916667],[0.80887392*param_scalar,0.85]]) # initial values for means of bins at predetermined speeds
    m = 'Re-initialization'#'Normal'##['Normal','Linear Adjustments',, 'Gaussian Adjustments'] #]#, 'Re-init sigma'] #'Gaussian Adjustments']#
    bins = np.array([0.9, 1.24, 1.42, 1.75])#np.array([-np.inf, 1.24, 1.42, np.inf])
    cur_bin = 0
    init_sigma_val = 0.08
    num_gens_cma = 50 # upper bound -- not predetermined
    meas_noise = 0.0
    spd_bins = len(bins)-1
    sigma = init_sigma_val*np.ones(spd_bins)
    λ, constants = init_constants(N, num_gens_cma)
    skip_conds = np.zeros((spd_bins,λ), dtype=float)
    param_bounds = np.zeros((N,2))
    #print("Max torq bound is:", torq_bound)
    param_bounds[0,:] = np.array([0,torq_bound]) # peak torque, max 1 = # kg of participant
    param_bounds[1,:] = np.array([0.375,1.]) # rise time, max = 40, min = 10
    constants.append(param_bounds)
    constants.append(meas_noise)
    constants.append(meas_noise) # don't have a goal rn
    new_param = np.array([0.701, 54.59, 27.8/rise_scalar, 9.98])
    upd_flag = False
    x_mean = init_xmean(bins, N, f_params, np.array([0.8, 1.25, 1.7])) # starting param values
    bins[0] = -np.inf
    bins[-1] = np.inf
    plot_sig_data = np.zeros((spd_bins,num_gens_cma+1))#, runs))
    plot_sig_data[:,0] = sigma#np.expand_dims(sigma,axis=-1)
    plot_mean_data = np.zeros((spd_bins,num_gens_cma+1, N))#, runs))
    plot_mean_data[:,0,:] = x_mean#np.expand_dims(x_mean,axis=-1)
    skip=False
    step_spd = 0
    # setting up the classification model
    gen_dir = save_dir + 'gen_data/'
    modelname = 'lr_model.pkl'
    steps_per_cond = 22#23
    steps_remove = 6#7
    num_constants = 6
    num_signals = 2
    num_bins = 30
    new_bout = True # flag to start the next bout
    walking = True # True when walking, False when taking a break
    bout_counter = 0 # keep track of current condition
    cond_step_counter = 0 # keep track of steps in current cond
    cur_cond_data = np.zeros((steps_per_cond,num_signals*num_bins))
    cond_data = np.zeros((spd_bins, λ, num_constants+num_signals*num_bins))
    cond_data[:,:,0] = weight
    cond_data[:,:,1] = height
    step_speeds = np.zeros((spd_bins, λ, steps_per_cond))
    cond_steps = np.zeros(steps_per_cond)

    # setting up bout distribution
    bout_data = np.load('bout_data/bout_data_45_2.npy')
    bout_dur, bout_break, bout_speed_list, bout_command_list = bout_data
    min_dur = 25
    for i, dur in enumerate(bout_dur):
        if int(dur) < min_dur:
            bout_dur[i] = str(min_dur)

    try: # TODO make sure re-loading works
        print("Loading:", save_dir + 'cma_data.npy')
        cma_data = np.load(save_dir + 'cma_data.npy', allow_pickle=True)
        cond_data = np.load(save_dir + 'subj_data.npy', allow_pickle=False)
        new_param, bout_counter, bin_gen_params, plot_sig_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter = cma_data
        print("Gen count:", gen_counter, "Cond count:", cond_counter, "\nCurrent mean:\n", grabMeans(plot_mean_data, gen_counter)*np.expand_dims(np.array([weight, rise_scalar]),axis=0), "\nCurrent params:\n", bin_gen_params*np.expand_dims(np.expand_dims(np.array([weight, rise_scalar]),axis=0),axis=0))
        #for i in range(np.sum(gen_counter)):
        #    _ = paramGen(np.copy(bin_gen_params), bin_opt_vars, param_bounds, cur_bin, λ) # make the seed random values match up in case sim stops
    except:
        print("Starting new optimization.")
        bin_opt_vars = []
        for i in range(spd_bins):
            bin_opt_vars.append(init_opt_vars(x_mean[i,:], sigma[i], N))
        cond_counter = np.zeros((spd_bins), dtype=int)
        gen_counter = np.zeros((spd_bins), dtype=int)
        bin_gen_params = np.zeros((spd_bins, λ, N)) # stores the params for the conditions
        for i in range(spd_bins):
            bin_gen_params = paramGen(bin_gen_params, bin_opt_vars, param_bounds, i, λ)
        print("Current mean:\n", grabMeans(plot_mean_data, gen_counter)*np.expand_dims(np.array([weight, rise_scalar]),axis=0), "\nCurrent params:\n", bin_gen_params*np.expand_dims(np.expand_dims(np.array([weight, rise_scalar]),axis=0),axis=0))
        #bin_gen_params[:,0,:] = x_mean # initialize the params
    for i in range(spd_bins): # send the params to the main loop to set up starting conditions to interpolate
        sendParams(e, i, new_param, bin_gen_params[i,cond_counter[i],:], weight, rise_scalar)

    bout_start_time = time.time()
    #bout_dur[bout_counter] = 2000 # keep in one condition for testing # TODO REMOVE
    while np.sum(gen_counter) < gen_thresh: # bout_counter < len(bout_speed_list) and # keep doing bouts until finished
        if not motor_ready:
            if c.qsize() > 0: # new data to pull in
                new_data = c.get()[0]# check for message saying motor is ready
                if len(new_data) == 3 and new_data[0] == 1:
                    motor_ready = True
                    print("Exo ready, entering CMA mode.")
                    bout_start_time = time.time()
                    new_bout = True
            else:
                time.sleep(cma_delay)
            continue
        bout_time = time.time() # check timer for walking bout durations
        if walking and (bout_time - bout_start_time > int(bout_dur[bout_counter])) and cond_step_counter == 0:
            #play_audio(, audio) # prompt audio stop command
            e.put(['Stop'])
            print("Stop.\n\n")
            bout_start_time = bout_time
            walking = False

        # wait until end of break and make sure they are stopped walking before starting next bout
        elif new_bout == False and not walking and (bout_time - bout_start_time > float(bout_break[bout_counter])):
            bout_counter += 1
            if bout_counter == len(bout_speed_list):
                bout_counter = 0 # restart!
                print("Restarting bout counter!")
            new_bout = True
            walking = True

        # updating the gait data whenever available
        if new_bout: # start new exo condition
            new_bout = False
            cond_step_counter = 0
            #spd = 1.0#float(bout_speed_list[bout_counter])
            #cur_bin = np.where(spd > bins)[0][-1]
            #play_audio(bout_command_list[bout_counter], audio)
            e.put([bout_command_list[bout_counter]])
            #new_param = sendParams(e, new_param, bin_gen_params[cur_bin, cond_counter[cur_bin], :], weight, rise_scalar) # send message to exo
            # TODO keep track of current parameters for CMA?
            print("Starting new bout. Dur", bout_dur[bout_counter], 'Spd', bout_speed_list[bout_counter], 'Cmd:', bout_command_list[bout_counter])
            print("")
        else:
            if c.qsize() > 0: # new data to pull in
                new_data = c.get()[0] # pull new gait data in from exo
                if len(new_data) == 2:
                    gait_data, step_spd = new_data
                    cond_steps[cond_step_counter] = step_spd
                    cur_cond_data[cond_step_counter,:] = gait_data # add to cur_cond_data
                    cond_step_counter += 1
                elif len(new_data) == 1 and walking: # skip
                    skip = True # skip the current condition
                    print("Skipping condition. Last step speed:", np.round(step_spd))
                    cond_step_counter = steps_per_cond
                    cur_cond_data[:,:] = 0.0
                elif len(new_data) == 3:
                    motor_ready = False
                    print("Exo not ready, pausing CMA.")
                elif not walking:
                    print("Ignoring skip while stopping.")
                else:
                    print("Unexpected number of elements passed:", len(new_data), new_data)
                print("Step", cond_step_counter, "Cond %:", np.round(cond_step_counter/steps_per_cond*100.0), "Bout", bout_counter, "Bout %:", np.round((bout_time - bout_start_time)/float(bout_dur[bout_counter])*100.0,1))

                if cond_step_counter == steps_per_cond: # end of a condition
                    if skip:
                        avg_spd = step_spd
                        skip = False
                        skip_conds[cur_bin, cond_counter[cur_bin]] = 1.0
                    else:
                        avg_spd = np.mean(cond_steps[steps_remove:])
                    cur_bin = np.where(avg_spd > bins)[0][-1] # fix here
                    step_speeds[cur_bin, cond_counter[cur_bin], :] = cond_steps
                    cond_data[cur_bin, cond_counter[cur_bin],num_constants:] = np.mean(cur_cond_data, axis=0)
                    new_param = getParams(cur_bin, cond_counter, new_param, bin_gen_params, weight, rise_scalar)
                    #print("Cond data avg:", np.mean(cur_cond_data, axis=0)[:10], new_param)
                    cond_data[cur_bin, cond_counter[cur_bin],3:num_constants] = new_param[1:]
                    cond_data[cur_bin, cond_counter[cur_bin],2] = new_param[0]/weight
                    cond_step_counter = 0
                    cond_counter[cur_bin] += 1 # increase counter and send next params
                    if cond_counter[cur_bin] == λ:# if generation finishes update CMA stuff
                        cond_counter[cur_bin] = 0
                        gen_counter[cur_bin] += 1
                        pw1_data, pairwise_conds = label_pairwise_mat(cond_data[cur_bin, :, :]) # input is [ordersize, 66]
                        pw1, pw1_prob = classifyModel(modelname, pw1_data) # load classifier
                        est_ordering, overlap, order_scores = compute_ordering_from_pairs(skip_conds[cur_bin,:], pw1, pairwise_conds, int(λ), pw1_prob)
                        unorder_args = np.argsort(est_ordering)
                        unorder_scores = order_scores[unorder_args] # scores for the conditions in order they were tested in CMA
                        skip_conds[cur_bin,:] *= 0.0
                        bin_opt_vars[cur_bin] = cma_multi([constants, bin_opt_vars[cur_bin]], unorder_args, bin_gen_params[cur_bin,:,:]) # passing bin specific parameters to cma
                        if m == 'Re-initialization':
                            bin_opt_vars, upd_flag = reinit_bin_upd(bin_opt_vars, cur_bin, spd_bins, bins, N, init_sigma_val, upd_flag, m)
                        bin_gen_params = paramGen(bin_gen_params, bin_opt_vars, param_bounds, cur_bin, λ)# update parameters
                        plot_sig_data, plot_mean_data = upd_plot_data(bin_opt_vars, gen_counter, cur_bin, plot_sig_data, plot_mean_data, spd_bins, constants)
                        cma_data = [new_param, bout_counter, bin_gen_params, plot_sig_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter]
                        fc = str(len(os.listdir(gen_dir))//4) # divide by number of types of files being saved
                        np.save(gen_dir + 'subj_data_' + fc + '.npy', cond_data)
                        np.save(gen_dir + 'cost_gen_' + fc + '.npy', unorder_scores)
                        np.save(gen_dir + 'walk_speed_' + fc + '.npy', step_speeds)
                        np.save(gen_dir + 'cma_data_' + fc + '.npy', cma_data)# save opt data
                        np.save(save_dir + 'final_params.npy', grabMeans(plot_mean_data, gen_counter)) # save the final mean params separately to make it easier to load?
                        print("Gen finished:", gen_counter, "\n Completed gen in bin:", cur_bin, '\nPrev bin mean:', grabMeans(plot_mean_data, gen_counter, True)*np.expand_dims(np.array([weight, rise_scalar]),axis=0),"\nCur Means:\n", grabMeans(plot_mean_data, gen_counter)*np.expand_dims(np.array([weight, rise_scalar]),axis=0), "\nPrev params:\n", cond_data[cur_bin,:,2:6]*np.array([weight,1.0,1.0,1.0]), "\nCosts:\n", unorder_scores)
                        print("Prev sigma:", plot_sig_data[cur_bin,np.sum(gen_counter)-1], "\nNew sigma:", plot_sig_data[cur_bin,np.sum(gen_counter)],"\nNew params:\n", bin_gen_params[cur_bin,:,:])
                    else:
                        cma_data = [new_param, bout_counter, bin_gen_params, plot_sig_data, plot_mean_data, bin_opt_vars, cond_counter, gen_counter]
                    np.save(save_dir + 'subj_data.npy', cond_data)
                    np.save(save_dir + 'cma_data.npy', cma_data)# save opt data
                    print("Starting new condition. Cond counter:", cond_counter, "Bin:", cur_bin, "Gen_counter:", gen_counter, "Spd:", np.round(avg_spd,1))
                    new_param = sendParams(e, cur_bin, new_param, bin_gen_params[cur_bin, cond_counter[cur_bin], :], weight, rise_scalar) # send message to exo
        time.sleep(cma_delay)

    np.save(save_dir + 'final_params.npy', grabMeans(plot_mean_data, gen_counter)) # save the final mean params separately to make it easier to load?
    # send last stop command
    e.put(['Stop'])
    print("Stop.\n\n")
    time.sleep(1)
    e.put(['Stop'])
    new_means = grabMeans(plot_mean_data, gen_counter)
    for i in range(spd_bins): # send the final params to the main loop to set up validation conditions to interpolate
        sendParams(e, i, new_param, new_means[i,:], weight, rise_scalar)
    print("Finished optimization. Gens:", gen_counter, "Conds:", cond_counter)
