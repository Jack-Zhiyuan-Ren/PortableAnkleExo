#!/usr/bin/env python3
"""
    # example command: python3 main.py LEFT 1 0 1 test_subj
    # arguments:
        which leg (LEFT/RIGHT),
        enable_motor (0 false, 1 true),
        visualize plotting (0 for False, 1 for True),
        CMA mode (1 for yes) otherwise static,
        save directory
"""
import os
import time
import busio
import digitalio
import board
from mcp3008 import MCP3008
from gpiozero import CPUTemperature
import can
from exoHelper import exoUpdate, exoUnpack, interpParams, initEncoder, loadOffline, motorUpdate, getTemp, message, readSensors, updateCycle, checkToeOff, connectMotor, connectExos
from exoHelper import loadParams, zeroMotor, killMotor, stateMachine, updateLRN, swingvel, torqueProfile, desiredTorque, disconnectExos
import numpy as np
import signal
import sys
from scipy import signal as sg
from multiprocessing import Process, Queue
import vlc
import sounddevice

rate = 200 # Hz of the control and sensing loops
plot_rate = 4# Hz to update plots (faster is more comp burden)
ts_len = 4 # length of time to show in plots (s) times samples per second
window_s = 3 # number of plot windows to save to each file
period_temp = 1.0 # time (s) btwn temp readings
interp_assistance = True # TODO set this up
enable_motor = True
plotter = True # run script to log data and maybe plot
vis = False # plot live data
saving = True
cma = False
offline = False
leg = 'LEFT' # Determine which leg is being run -- changes child/parent relationship (start RIGHT leg pi first)
save_folder = 'rt' #'Untethered_dualleg_2' # name to save to
offline_folder = '/home/pi/exo/save/Untethered_dualleg_2/run_2/'
init_tau = 1.5 # Nm to target when zeroing motor
max_angle = 0.53 # 0.47 = 27deg, 0.523 = 30 # max angle (rad) of ankle plantar flexion
stand_angle_thresh = 0.05 # must reach this angle to break out of standing condition
stand_angle = 0.0
heelstrike_timing_adjust = 7/200. # 6/200 before, 7/200 shift the target phase by this amount (in seconds) due to delay in heelstrike detection
weight = 30 #68 # kg
height = 1.76 # meters
load_cma = False # if true load the cma finalized params in save folder as the starting parameters
audio = True # set to false to not have audio commands
ramp_steps = 5 #5 # number of steps to ramp up torque over
max_torq = 53 # max exo can do is 55 Nm, make sure we don't try params above that
static_bin = 1
gen_thresh = 24
fast_stance = 0.96 # 1.02 #0.86 #0.99 # stance time at 1.7 m/s (3.6 treadmill)
slow_stance = 1.2 #1.37#1.28 #1.36 #1.1 #1.25 # (1.8 on treadmill)
velmdes_negthresh = -20.0 # max motor command for neg velocity
def set_commandline(args, default):
    bool_list = [2,3,7,9]
    float_list = [4,5,6,12,13]
    int_list = [8,10,11]
    for i, arg in enumerate(args[1:]):
        if i in bool_list:
            if arg == 'True':
                arg = True
            else:
                arg = False
        elif i in float_list:
            arg = float(arg)
        elif i in int_list:
            arg = int(arg)
        default[i] = arg
    return default

# script example: python3 main.py save_folder LEFT enable_motor cma height(m) weight(kg) max_angle(rad) load_cma(bool)
if len(sys.argv) > 1:
    default = [save_folder, leg, enable_motor, cma, height, weight, max_angle, load_cma, ramp_steps, interp_assistance, static_bin, gen_thresh, slow_stance, fast_stance]
    new_params = set_commandline(sys.argv, default)
    save_folder, leg, enable_motor, cma, height, weight, max_angle, load_cma, ramp_steps, interp_assistance, static_bin, gen_thresh, slow_stance, fast_stance = new_params
    print(new_params)
save_dir = '/home/pi/exo/save/' + save_folder + '/'

if leg == 'RIGHT':
    audio = False
if audio:
    instance = vlc.Instance()
    player = instance.media_player_new()
    path_audio = '/home/pi/exo/sound_files/'

if not cma:
    _,_,_,val_list = np.load('/home/pi/exo/bout_data/bout_data_val.npy')
    val_ind = 0

# TODO add load feature for other interp params
rise_scalar = 40.0
torq_bound = min(max_torq, weight)
interp_params = np.array([[min(torq_bound, weight*0.64051748), 54.59, 0.58666667*rise_scalar, 9.98],[min(torq_bound, weight*0.72715882), 54.59, 0.70916667*rise_scalar, 9.98],[min(torq_bound, weight*0.80887392), 54.59, 0.85*rise_scalar, 9.98]])
# GA_params = np.array([[],[],[]])
incl_params = np.array([min(torq_bound, weight*0.84887392), 54.59, 0.758*rise_scalar, 9.98]) #0.608*rise_scalar, 9.98])
# pat fast speed opt for incline: np.array([min(torq_bound, weight*0.816), 54.59, 0.92*rise_scalar, 9.98])
# nw  = #0.82584698, 0.81831322, fw = 0.825, 0.92
static_params = np.array([min(torq_bound, weight*0.82584698), 54.59, 0.92*rise_scalar, 9.98]) #np.array([min(torq_bound, weight*0.816), 54.59, 0.92*rise_scalar, 9.98]) #np.array([min(torq_bound, weight*0.72715882), 54.59, 0.70916667*rise_scalar, 9.98]) #np.array([10,54.8,33.4,9.1]) #interp_params[1,:]#[41, 53.6, 24.1, 10.0]# #[46.1,54.8,33.4,9.1] -- pat DD parameters
speed_params = np.array([9.58239369128718,-24.2954102910252,16.2100069156388]) # for converting gait duration into m/s
torq_spds = np.array([0.0, 1.07, 1.33, 1.58, 1.8])

if load_cma:
    gen_means = np.load(save_dir + 'cma/final_params.npy')
    for i in range(3):
        interp_params[i,:] = loadParams(interp_params[i,:], gen_means[i,:], weight, rise_scalar)
    print("Loading cma opt params:", interp_params)
    # TODO load as interp params

if not interp_assistance:
    if static_bin > 2:
        static_params = incl_params # params for incline
    else:
        static_params = interp_params[static_bin, :]
    print("Constant assistance:", static_params)

if offline:
    data = loadOffline(offline_folder)
    off_cnt = 0

# defining variables
k_gain = 5.0 # velocity gain sent to CubeMars controller (max 5)
kt = 7.0/k_gain # 8.0 for untethered prev #3.5 for static, 8.5 got spiky for 35 Nm
kv = 0.3/k_gain #1.0/k_gain # 0.5 for static
KL = 0.6/k_gain # 0.4 to 0.6 works well
D = 4 # 4 is best # 3 to 5 all about the same compensating for delay in IL
phase_cutoff = 80.0
end_phase = 100.0
dt = 1./rate
future_time = 0.0*dt # 5 or 6 # how many steps ahead the desired torque should be (to account for motor delay)

# motor settings
thresh_strike = 0.7
thresh_mid = 0.6 #1.3 # thresh mid has to be higher than thresh_launch
thresh_launch = 0.9 #2.0
steps_before_torq = 2
num_strike = 3 # times in a row pressure sensors meet specs to heelstrike
num_launch = 3 # times in a row pressure sensors meet specs to launch
mu_str = 1.0 # 0.9 # weighting of old stance times # for SS walking
tau_ramp = 0.0 # scalar to multiple by
tswing_min = 0.3 # minimum swing time in seconds
to_pct = 0.50 # minimum time in % gait before toe off
to_spds = np.array([0.0, 1.0, 1.3, 1.7, 1.8])
to_vals = np.array([0.63, 0.61, 0.56, 0.48, 0.48]) # should be 66, 63, 51 based on paper
heelstrike_ind = 0.0 # send walk speed to cma when new step detected, otherwise send 0
tstance_pct = min(to_vals) #55 # minimum stance time in % gait
tstride_avg_init = 0.9 #1.2
m2 = (1.6 - 0.8133)/(fast_stance - slow_stance)
offset2 = 0.8133 - m2*slow_stance
#print("Expected slow speed for current parameters is", np.round(slow_stance*m2+offset2,2), "m/s (should be around 0.813)")
num_prev_steps = 5
prev_times = np.zeros(num_prev_steps)
walk_speed = np.zeros(num_prev_steps)
max_step_time = 2.0 # reset gaits if longer than this many seconds
N_consecutive_steps = 0
motor_connected = np.copy(enable_motor) # constant just to see if it was originally turned on or not to re-turn on
check1 = False
check3 = False
check4 = False
check5 = False
prev_ang_flag = False
stand = True
stand_thresh = 1.45 #1.35#1.22 # time in seconds -- maybe 1.2s if not cutting off soon enough?. 1.4 cut off too early.
decel_thresh = -0.8

can_button = False
drum_ratio = 5 #0.115/0.023
slack = 0.023/(2*3.14159*0.02)*drum_ratio # slack in meters (times drum radius(0.2) for dist)
b2,a2 = sg.butter(N=2, Wn=10, fs = rate, analog=False) # 3rd order 20hz butter
b1,a1 = sg.butter(N=2, Wn=20, fs = rate, analog=False) # 3rd order 20hz butter

# setting up
path = save_folder
if saving:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + 'cma/')
        os.makedirs(save_dir + 'cma/gen_data/')
        run = 1
    else:
        run = len(os.listdir(save_dir))
        print("Starting run", run)
    path = save_dir + 'run_' + str(run) + '/'
    os.makedirs(path) # creating save directory
if cma and leg == 'RIGHT':
    cma = False # only run CMA on left leg!!!

if plotter or vis: # starting up plot and save thread
    import exoPlotter as plott
    p = Queue() # queue for button messages
    c = Queue() # queue for cma messages
    d = Queue() # queue for passing cma messages from main to cmaThread
    e = Queue() # param messages from cma to exo
    if vis:
        plot = Process(target=plott.plotting, args=(saving, path, vis, p, c, rate, plot_rate, ts_len, window_s, cma), daemon=True)
    else:
        plot = Process(target=plott.createData, args=(saving, path, vis, None, p, c, rate, plot_rate, ts_len, window_s, cma), daemon=True)
    plot.start() # spawning IMU process

if cma:
    import cmaThread as cmat
    tb2 = torq_bound/weight
    cma_thread = Process(target=cmat.cma, args=(save_dir+'cma/', gen_thresh, height, weight, d, e, tb2, rise_scalar), daemon=True)
    cma_thread.start()

# setup sensors
exit_flag = False
adc = MCP3008(bus=1,device=0)
yp = np.flip(np.array([15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150])) # resistances at temps
xp = np.flip(np.array([1.571,1.249,1,0.8057,0.6531,0.5327,0.4369,0.3603,0.2986,0.2488,0.2083,0.1752,0.1481,0.1258,0.1072,0.09177,0.07885,0.068,0.05886,0.05112,0.04454,0.03893,0.03417,0.03009,0.02654,0.02348,0.02083,0.01853])) # corresponding temps)
button = digitalio.DigitalInOut(board.D24)
button.switch_to_input(pull=digitalio.Pull.DOWN)
skip_button = digitalio.DigitalInOut(board.D12)
skip_button.switch_to_input(pull=digitalio.Pull.DOWN)
mot_kill = digitalio.DigitalInOut(board.D5)
mot_kill.switch_to_input(pull=digitalio.Pull.DOWN)
cpu = CPUTemperature()
# Initialize storage and variables
start_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC]
stop_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFD]
zero_position = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE]
exo_id = 0x25
m_id = 0x01 # motor ID
wait_time = 0.001 #0.0005
mpos = 0.0
mvel = 0.0
mvel_prev = 0.0
prev_mvel = 0.0
mtorq = 0.0
mot_temp = 0.0
cpu_temp = 0.0
prev_cpu_temp = 0.0
des_torq = 0.0
phase = 0.0
pr = np.zeros(4)
stance = False
prev_button = False
button_thresh = 0.8 # delay in seconds before button can be pressed again
skip_stand_thresh = 2.5 # seconds
button_mod = 20 # divide rate by this for how often button is checked
cntl_mode = 0 # 0=REST, 1=SETZERO, 2=CONSTTAU, 3=CONSTPOS, 4=WALK
torq_prof = torqueProfile(static_params, torq_bound)
ank_offset = initEncoder(adc)
window = ts_len*rate*window_s
short_window = (ts_len//2)*rate
torq_vec = np.zeros(window)
pos_vec = np.zeros(window)
pos_filt = np.zeros(ts_len*rate)
torq_filt = np.zeros(ts_len*rate)
skip = False
last_skip_button_time = 0.0
prev_skip = False
vel = 0.0
des_future_torq = 0.0
pos_comp = 0.0 # pos component of torque
vel_comp = 0.0 # velocity component of torque
il_comp = 0.0 # IL component of torque
ankle_pos = 0.0 # ankle angle in radians
meas_torq = 0.0 # heelspur torque measurement
mpos_offset = 0.0 # offset position for calibration
velmdes = 0.0 # commanded motor velocity
# IL params
max_cnt_lrn = rate*20
dataLRN = np.zeros((max_cnt_lrn+1,2))
# control gains
init_tau_vel = 0.1*drum_ratio
k_zt = 100.0/k_gain # 120.0
# can also tune KL gain above
beta = 0.97 # Forgetting term -- prev had 0.95, previously had it at 0.99 -- changing to 0.9
mu = 1.0
kp = 1.0/k_gain
kd = 0.05/k_gain
err_limit = 25 # print commanded velocity if above this
ctrl_mode = 0 # state of device (0 = rest, 1 = const_tau, 2 = zero-torque, 3 = static torq, 4 = cma torq)
gait_phase = 0 # 0 for stance, 1 for swing

# send or look for CAN message to link the exos
can_exo = connectExos(leg, start_motor_mode, wait_time, exo_id)

if enable_motor: # initializing and connecting motor
    initialized, _, can0, mpos, mvel, mtorq = connectMotor(m_id, start_motor_mode, wait_time, zero_position, mpos, mvel, mtorq)
else:
    initialized = False
    can0 = None

def exit_function(signal, frame):
    print('Exiting...')
    if enable_motor:
        killMotor(can0, m_id, stop_motor_mode) # disable motor mode
    disconnectExos()
    if plotter: plot.terminate()
    if cma: cma_thread.terminate()
    time.sleep(1.0)
    sys.exit(0)
signal.signal(signal.SIGINT, exit_function) # init exit

start_time = time.time() # time since last main loop entry
temp_time = start_time # time since last grab of temp data
int_time = start_time # time since last print
init_time = start_time # time since start of program
last_step_time = start_time # time since last heelstrike
last_button_time = start_time
cnt = 1 # number of times in main control loop
main_cnt = 1
print_int = int(0.9*rate+1) # divide rate by print int
initial_motor_flag = True
start_stand = False
launch_ready = False
strike_avg = 0
launch_avg = 0
tstride_avg = tstride_avg_init
tstride = 0.0
tstance = 0.0
istride = 0
istance = 0
iswing = 0
Nstrides = 0


while(not exit_flag): ### main control and sensing loop
    cur_time = time.time()
    if (cur_time - start_time) >= dt: # wait for sample time
        if offline: # load fake data
            vec = data[:,off_cnt]
            p.put([vec[3:7], vec[1], vec[8], vec[10], vec[2], vec[15], vec[16], vec[9], vec[11], vec[0], vec[7], vec[11], 3, False, vec[14], vec[12], vec[17], vec[18]])
            off_cnt += 1
            if e.qsize() > 0:
                cur_bin, new_param = e.get()
            if off_cnt == data.shape[1]:
                break
            if c.qsize()>0:
                d.put([c.get()])
            start_time = cur_time # leave in to run in real-time, otherwise run as fast as possible
            continue

        # rest of code for real-time data
        start_time = cur_time # reset timer
        if cma:
            if e.qsize() > 0:
                data = e.get()
                if len(data) == 1:
                    text_to_audio = data[0]
                    if audio:
                        source = path_audio + text_to_audio + '.mp3'
                        media = instance.media_new(source)
                        player.set_media(media)
                        player.play()
                    #play_audio(text_to_audio, audio)
                else:
                    cur_bin, interp_params[cur_bin,:] = data # updating static params for new CMA params
                    print("New params:",interp_params[cur_bin,:], cur_bin) # add to the current
                    exoUpdate(can_exo, exo_id, interp_params[cur_bin,0], interp_params[cur_bin,2], cur_bin, wait_time) # send to right leg
            if c.qsize()>0:
                    d.put([c.get()])
        elif leg == 'RIGHT': # check for messages over CAN
            button_msg = can_exo.recv(wait_time)
            if button_msg != None:
                if list(button_msg.data) == stop_motor_mode: # update state machine based on button press
                    can_button = True
                    prev_button = True
                elif list(button_msg.data) == zero_position: # exit program
                    exit_flag = True
                elif list(button_msg.data) == start_motor_mode:
                    skip = True
                    last_skip_button_time = cur_time
                    N_consecutive_steps = 0
                    stand = True
                else:
                    # process code to get the new parameters and cur_bin to update other exo controller
                    interp_params, cur_bin = exoUnpack(interp_params, button_msg)
                    print("Updated right leg params:", interp_params[cur_bin,:], cur_bin)


        cnt += 1
        main_cnt += 1
        sensor_data = readSensors(adc, leg, pr, stance, ank_offset, thresh_strike) # get data from sensors

        print("sensor data: " , sensor_data[2])
        strike_avg, launch_avg = updateCycle(sensor_data[2], strike_avg, launch_avg, thresh_strike, thresh_launch)
        
        ### updating and filtering sensor data
        if cnt >= ts_len*rate: # filter and control once enough data to filter
            if cnt%window==0:
                cnt = ts_len*rate
                torq_vec[:ts_len*rate] = torq_vec[-ts_len*rate:]
                pos_vec[:ts_len*rate] = pos_vec[-ts_len*rate:]
            pos_vec[cnt] = sensor_data[1]  #ankle_position
            torq_vec[cnt] = sensor_data[0] # measured torque
            pos_filt = sg.filtfilt(b2, a2, pos_vec[cnt-ts_len*rate:cnt], method='gust', irlen=100) # 134
            ankle_pos = pos_filt[-1]
            vel = (ankle_pos - pos_filt[-2])*rate
            torq_filt = sg.filtfilt(b1, a1, torq_vec[cnt-ts_len*rate:cnt], method='gust', irlen=100) # 69
            meas_torq = torq_filt[-1]

            if initial_motor_flag and ctrl_mode >= 3: # start init_time to not jump into cycle somewhere random
                init_time = cur_time
                initial_motor_flag = False
            phase, des_torq, des_future_torq = desiredTorque(tau_ramp, heelstrike_timing_adjust, torq_prof, tstride_avg, istride*dt, init_tau, future_time) # determine desired torque

            if meas_torq < -5.0 and enable_motor: # error with torque measurements on heelspur
                print("Error with heelspur torque measurements.")
                enable_motor, initialized = killMotor(can0, m_id, stop_motor_mode)

            ### Motor control architecture
            if ctrl_mode == 0: # REST
                velmdes = 0
                des_torq = 0
                tstride_avg = tstride_avg_init
                N_consecutive_steps = 0
                gait_phase = False
                stand = True

            elif ctrl_mode == 1: # CONST_TAU -- calibrate motor cable
                if not initialized and motor_connected:
                    initialized, enable_motor, can0, mpos, mvel, mtorq = connectMotor(m_id, start_motor_mode, wait_time, zero_position, mpos, mvel, mtorq)
                velmdes = init_tau_vel
                des_torq = init_tau
                if meas_torq > init_tau and enable_motor: # calibration finished
                    mpos, mvel, mtorq = zeroMotor(can0, m_id, zero_position, mpos, mvel, mtorq, wait_time) # zero motor position
                    mpos_offset = ankle_pos*drum_ratio
                    ctrl_mode = 2
            elif ctrl_mode >= 2: # WALK
                # limit desired motor velocity and position to acceptable range?
                if gait_phase: # swing
                    istride += 1
                    istance = 0
                    iswing += 1
                    launch_ready = False
                    if ctrl_mode == 2 or stand:
                        velmdes = swingvel(ankle_pos,vel,mpos,drum_ratio,k_zt,dt,slack,mpos_offset)
                    else:
                        velmdes = swingvel(ankle_pos,vel,mpos,drum_ratio,k_zt,dt,slack,mpos_offset, phase, phase_cutoff, end_phase)
                    if strike_avg >= num_strike and iswing*dt > tswing_min: # and not check4: # look for heelstrike
                        gait_phase = False # switch to stance
                        last_step_time = cur_time
                        print("STANCE", strike_avg)
                        tstride = istride*dt
                        Nstrides += 1
                        #if tstride < 1.8*tstride_avg and tstride > 0.5*tstride_avg: # maybe include Nstrides to
                        tstride_avg = tstride_avg*(1-mu_str) + tstride*mu_str
                        istride = 0
                        iswing = 0
                        if tstride >= max_step_time or (cur_time - last_step_time) >= max_step_time:
                            prev_times = 0*prev_times
                            walk_speed = 0*walk_speed
                            N_consecutive_steps = 0
                            tstride_avg = tstride_avg_init
                        else:
                            prev_times[1:] = prev_times[:-1]
                            walk_speed[1:] = walk_speed[:-1]
                            prev_times[0] = tstride
                            if tstride >= 1.25: # cutoff for linear approx -- they both match 0.8 m/s there
                                walk_speed[0] = max(0, 1.9383-0.9*tstride) # 2.025 offset normally
                            else:
                                walk_speed[0] = min(1.8, offset2 + m2*tstride)
                                #walk_speed[0] = np.polyval(speed_params, tstride) # TODO pick a better model -- make linear for lower speeds?
                            clipped_spd = min(max(walk_speed[0], torq_spds[0]), torq_spds[-1])

                            if interp_assistance: # TODO: interpolate desired torque based on previous walking speed
                                static_params = interpParams(interp_params, torq_spds, clipped_spd, torq_bound)
                                torq_prof = torqueProfile(static_params, torq_bound)
                                to_pct = np.interp(clipped_spd, to_spds, to_vals) # TODO: interpolate max stance time based on previous walking speed
                                #print("Interp to_pct:", walk_speed[0], to_pct, to_spds, to_vals)
                                #print("Interp assistance:", walk_speed[0], static_params, interp_params)
                            if ctrl_mode !=2:
                                N_consecutive_steps += 1
                        tau_ramp = min((N_consecutive_steps+1)/ramp_steps, 1.0)
                else: # stance phase - perform torque control
                    istride += 1
                    istance += 1
                    iswing = 0
                    if ctrl_mode == 2: # zero torque
                        velmdes = swingvel(ankle_pos,vel,mpos,drum_ratio,k_zt,dt,slack,mpos_offset)
                        check4 = False
                    elif ctrl_mode == 3: # static torque
#                         if N_consecutive_steps < steps_before_torq:
#                             des_torq = 0
#                             des_future_torq = 0
                        if des_future_torq < init_tau:
                            if (istance*dt) < tstance_pct*tstride_avg: # during stance
                                des_future_torq = init_tau
                                des_torq = init_tau
                            else: # switch to zero-torque
                                des_torq = 0

                        if stand or N_consecutive_steps < steps_before_torq or (ankle_pos > max_angle):
                            velmdes = swingvel(ankle_pos,vel,mpos,drum_ratio,k_zt,dt,slack,mpos_offset)
                        else:
                            pos_comp = (des_future_torq - meas_torq)*kt
                            vel_comp = -mvel*kv
                            il_comp = dataLRN[min(istride, max_cnt_lrn-D) + D, 0]
                            velmdes = pos_comp + vel_comp + il_comp
                    elif ctrl_mode == 4: # CMA TORQ
                        pass

            if not gait_phase and ctrl_mode > 2: # iterative learning update
                dataLRN = updateLRN(istride, max_cnt_lrn, torq_filt, des_future_torq, dataLRN, mu, beta, KL)
            # SAFETY CONDITIONS
            check1 = ((istance*dt) >= to_pct*tstride_avg) and (sensor_data[2][0] > thresh_mid) and (N_consecutive_steps > 2) # check this meets minimum toe off time and heel not on ground
            check3 = (istance*dt) > tstance_pct*tstride_avg
            check4 = False#(ankle_pos > max_angle)
            check5, launch_ready = checkToeOff(sensor_data[2],thresh_mid,launch_ready,launch_avg,num_launch)

            if check1 or (check3 and check5):# or check4: # switch to swing # check1 or
                gait_phase = True
                launch_ready = False
                #if not prev_ang_flag:
                print("SWING. Time flag:", check1, np.round(to_pct,2), 'Launch flag:', (check3 and check5), 'Angle flag:', check4, 'Launch/stance cnt:', launch_avg, strike_avg)
                    #if check4:
                    #    prev_ang_flag = True
                if istance*dt < stand_thresh and (N_consecutive_steps >= 1) and stand_angle > stand_angle_thresh and ((cur_time - last_skip_button_time) > skip_stand_thresh): # once a step is shorter than that thresh then start applying torques.
                    if stand:
                        print("END STAND.")
                    stand = False
            #else:
            #    prev_ang_flag = False
            if stand: #(cur_time - last_stand_time) < stand_time_thresh: # Add a case to reset the angle once fully stopped?
                if ankle_pos > stand_angle:
                    stand_angle = ankle_pos
                if np.std(pos_filt[short_window:]) < 0.033:
                    N_consecutive_steps = 0
                    stand_angle = 0
            elif istance*dt >= stand_thresh:
                if not start_stand:
                    start_stand = True
                    print("Stand thresh reached")
                stand = True
                #last_stand_time = cur_time
                N_consecutive_steps = 0
                stand_angle = 0.0
            elif (N_consecutive_steps > 2) and np.mean(walk_speed[:2] - walk_speed[1:3]) < decel_thresh and (2*prev_times[0] - prev_times[1]) >= stand_thresh: # trend from previous step is too slow
                stand = True
                #last_stand_time = cur_time
                N_consecutive_steps = 0
                stand_angle = 0.0
                # des_torq = 0
                if not start_stand:
                    start_stand = True
                    last_skip_button_time = cur_time
                    print("Previous step trending slow:", prev_times[0], "3 stp change:", np.mean(prev_times[:2] - prev_times[1:3]), -0.15, np.round(prev_times[0]-prev_times[1],2), stand_thresh, 'Spd:', walk_speed[0])
#             elif np.std(pos_filt) < 0.032: # no movement so
#                 stand = True
#                 if not start_stand:
#                     start_stand = True
#                     print("No motion, detect stand.")
            else:
                start_stand = False

            if enable_motor: # command torque and get motor data
                if velmdes <= velmdes_negthresh:
                    velmdes = velmdes_negthresh
                #    print("clipping", velmdes)
                cmd_arr = [0.0, velmdes, 0.0, k_gain, 0.0]
                mvel_prev = mvel
                mpos, mvel, mtorq = motorUpdate(can0, m_id, cmd_arr, mpos, mvel, mtorq, wait_time)
        if plotter: ### send data to plotter / opt / save
            #sensor_data[2] = np.array([pos_comp, vel_comp, il_comp, velmdes/2.0])
            if istride == 1:
                heelstrike_ind = walk_speed[0]
            else:
                heelstrike_ind = 0.0
            p.put([sensor_data, des_torq, mpos/drum_ratio, mvel/drum_ratio, mtorq, mot_temp, cpu_temp, ankle_pos, vel, meas_torq, phase, cur_time-init_time, ctrl_mode, enable_motor, velmdes, not gait_phase, il_comp, heelstrike_ind])
        if main_cnt%print_int == 0: # serial print control loop rate
            ctrl_rate = 1/((cur_time - int_time)/print_int)
            int_time = cur_time # print_int
            print("Rate:",np.round(ctrl_rate,1),"Hz, Mins:",int((cur_time - init_time)/60.0),"Mode:", ctrl_mode, "CPU:", int(prev_cpu_temp), "Swing:", gait_phase, 'Spd:', np.round(walk_speed[0],2), "Stand:", stand, 'Steps:', N_consecutive_steps, 'Dur', np.around(prev_times[:4],2), "launch/strike:", launch_avg, strike_avg, "Torq:", np.round(meas_torq,1), 'P:', np.around(static_params[:-1],1), 'Ang:', np.round(ankle_pos,2), 'to:', np.round(to_pct,2))#, np.std(pos_filt))
            if ctrl_rate < 0.7*rate and enable_motor: # error in control loop keeping up
                enable_motor, initialized = killMotor(can0, m_id, stop_motor_mode)
                print("Control loop rate error:", np.round(ctrl_rate,1))
                print(velmdes, mtorq, mpos, mvel, mvel_prev, des_torq, gait_phase, np.round(tstride_avg,2), strike_avg, launch_avg)
        if main_cnt%button_mod == 0: # check exit and motor kill buttons
            if (mot_kill.value or can_button) and (cur_time - last_button_time > button_thresh):
                if prev_button:
                    last_button_time = cur_time
                    can_button = False
                    if leg == 'LEFT':
                        can_exo.send(message(exo_id, stop_motor_mode))
                prev_button, enable_motor, ctrl_mode, initialized = stateMachine(d, ctrl_mode, enable_motor, prev_button, can0, m_id, stop_motor_mode, initialized)
            else:
                prev_button = False
            if leg == 'LEFT':
                exit_flag = button.value
                if exit_flag:
                    can_exo.send(message(exo_id, zero_position))
        if main_cnt%(button_mod+1) == 0 and leg == 'LEFT': # check skip button
            if (skip_button.value) and (cur_time - last_skip_button_time > skip_stand_thresh):
                if prev_skip_button:
                    last_skip_button_time = cur_time
                    skip = True
                    print("SKIP BUTTON PRESSED")
                    stand = True
                    can_exo.send(message(exo_id, start_motor_mode))
                    N_consecutive_steps = 0
                    if cma:
                        d.put([[skip]])
                    else: # play bout command
                        if val_ind < len(val_list):
                            if audio:
                                source = path_audio + val_list[val_ind] + '.mp3'
                                media = instance.media_new(source)
                                player.set_media(media)
                                player.play()
                            #play_audio(val_list[val_ind], audio)
                            val_ind += 1
                        else:
                            if audio:
                                source = path + 'Stop' + '.mp3'
                                media = instance.media_new(source)
                                player.set_media(media)
                                player.play()
                            #play_audio('Stop', audio)
                    prev_skip_button = False
                else:
                    prev_skip_button = True
                #prev_button, enable_motor, ctrl_mode, initialized = stateMachine(ctrl_mode, enable_motor, prev_button, can0, m_id, stop_motor_mode, initialized)
            else:
                prev_skip_button = False
                skip = False
        elif leg == 'RIGHT':
            if skip and (cur_time - last_skip_button_time > skip_stand_thresh):
                skip = False
        if (cur_time - temp_time) >= period_temp: # temp sensor
            temp_time = cur_time
            mot_temp, cpu_temp = getTemp(adc, cpu, xp, yp)
            prev_cpu_temp = cpu_temp
        else:
            cpu_temp = 0.0

### EXITING
if enable_motor:
    killMotor(can0, m_id, stop_motor_mode)
disconnectExos()
if plotter: plot.terminate()
if cma: cma_thread.terminate()
print("Program ending...")
time.sleep(1.0)
sys.exit(0)
