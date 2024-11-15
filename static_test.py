# run this to have just a simple torque tracking controller in the static test rig
import os
import time
import busio
import digitalio
import board
from mcp3008 import MCP3008
from gpiozero import CPUTemperature
import can
from exoHelper import initEncoder, motorUpdate, getTemp, message, readSensors, updateCycle, checkToeOff, connectMotor, connectExos
from exoHelper import zeroMotor, killMotor, stateMachine, updateLRN, swingvel, torqueProfile, desiredTorque, disconnectExos
import numpy as np
import signal
import sys
from scipy import signal as sg

rate = 200 # Hz of the control and sensing loops
plot_rate = 4# Hz to update plots (faster is more comp burden)
ts_len = 4 # length of time to show in plots (s) times samples per second
window_s = 3 # number of plot windows to save to each file
torq_lim = 5. # software max forf torque applied NOT USED RIGHT NOW
period_temp = 1.0 # time (s) btwn temp readings
enable_motor = True
step_mode = True # if step mode then apply a step of 10*init_tau -- otherwise static tracking
plotter = True # run script to log data and maybe plot
vis = False # plot live data
saving = False
cma = False
torq_bound = 40.0 # max Nm
leg = 'LEFT' # Determine which leg is being run -- changes child/parent relationship (start RIGHT leg pi first)
save_folder = 'new_motor_test' # name to save to
init_tau = 2.0 # Nm to target when zeroing motor
max_angle = 0.47 # max angle (rad) of ankle plantar flexion
heelstrike_timing_adjust = 7/200. # shift the target phase by this amount (in seconds) due to delay in heelstrike detection
static_params = [10.0,54.5,33.4,9.1] #[46.1,54.8,33.4,9.1] -- pat DD parameters

# defining variables
k_gain = 4.0 # velocity gain sent to CubeMars controller (max 5)
kt = 3.5/k_gain # 3.5 for static, 8.5 got spiky for 35 Nm
kv = 0.5/k_gain#1.0/k_gain # 0.5 for static
KL = 0.00/k_gain
D = 4 #8, 4 are best #14 #10 #4 #15 compensating for delay in IL
phase_cutoff = 80.0
end_phase = 100.0
dt = 1./rate
future_time = 5.0*dt # 5 or 6 # how many steps ahead the desired torque should be (to account for motor delay)

# motor settings
thresh_strike = 0.7
thresh_mid = 1.3
thresh_launch = 2.0
num_strike = 3 # times in a row pressure sensors meet specs to heelstrike
num_launch = 3 # times in a row pressure sensors meet specs to launch
mu_str = 0.9 # weighting of old stance times # for SS walking
ramp_steps = 10 # number of steps to ramp up torque over
tau_ramp = 1.0 # scalar to multiple by
tswing_min = 0.3 # minimum swing time in seconds
to_pct = 0.61 # minimum time in % gait before toe off
tstance_pct = 0.55 # minimum stance time in % gait
num_prev_steps = 5
prev_times = np.zeros(num_prev_steps)
max_step_time = 2.4 # reset gaits if longer than 3s
N_consecutive_steps = 0
motor_connected = np.copy(enable_motor) # constant just to see if it was originally turned on or not to re-turn on
check1 = False
check3 = False
check4 = False
check5 = False
can_button = False
drum_ratio = 5 #0.115/0.023
slack = 0.02/(2*3.14159*0.02)*drum_ratio # slack in meters (times drum radius(0.2) for dist)
b2,a2 = sg.butter(N=2, Wn=10, fs = rate, analog=False) # 3rd order 20hz butter
b1,a1 = sg.butter(N=2, Wn=20, fs = rate, analog=False) # 3rd order 20hz butter

# setting up
path = save_folder
if saving:
    save_dir = '/save/' + save_folder + '/'
    # save_dir = '/home/pi/Desktop/exo/save/' + save_folder + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        run = 1
    else:
        run = len(os.listdir(save_dir))+1
        print("Starting run", run)
    path = save_dir + 'run_' + str(run) + '/'
    os.makedirs(path) # creating save directory

if plotter or vis: # starting up plot and save thread
    from multiprocessing import Process, Queue
    import exoPlotter as plotter
    p = Queue() # queue for button messages
    if vis:
        plot = Process(target=plotter.plotting, args=(saving, path, vis, p, rate, plot_rate, ts_len, window_s, torq_lim), daemon=True)
    else:
        plot = Process(target=plotter.createData, args=(saving, path, vis, None, p, rate, plot_rate, ts_len, window_s, torq_lim), daemon=True)
    plot.start() # spawning IMU process
# setup sensors
exit_flag = False
adc = MCP3008(bus=1,device=0)
yp = np.flip(np.array([15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150])) # resistances at temps
xp = np.flip(np.array([1.571,1.249,1,0.8057,0.6531,0.5327,0.4369,0.3603,0.2986,0.2488,0.2083,0.1752,0.1481,0.1258,0.1072,0.09177,0.07885,0.068,0.05886,0.05112,0.04454,0.03893,0.03417,0.03009,0.02654,0.02348,0.02083,0.01853])) # corresponding temps)
button = digitalio.DigitalInOut(board.D24)
button.switch_to_input(pull=digitalio.Pull.DOWN)
mot_kill = digitalio.DigitalInOut(board.D5)
mot_kill.switch_to_input(pull=digitalio.Pull.DOWN)
cpu = CPUTemperature()
# Initialize storage and variables
start_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC]
stop_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFD]
zero_position = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE]
exo_id = 0x25
m_id = 0x01 # motor ID
wait_time = 0.0001
mpos = 0.0
mvel = 0.0
prev_mvel = 0.0
mtorq = 0.0
mot_temp = 0.0
cpu_temp = 0.0
des_torq = 0.0
phase = 0.0
pr = np.zeros(4)
stance = False
prev_button = False
button_thresh = 0.8 # delay in seconds before button can be pressed again
button_mod = 20 # divide rate by this for how often button is checked
cntl_mode = 0 # 0=REST, 1=SETZERO, 2=CONSTTAU, 3=CONSTPOS, 4=WALK
torq_prof = torqueProfile(static_params, torq_bound)
ank_offset = initEncoder(adc)
window = ts_len*rate*window_s
torq_vec = np.zeros(window)
pos_vec = np.zeros(window)
pos_filt = np.zeros(ts_len*rate)
torq_filt = np.zeros(ts_len*rate)
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
beta = 0.99
mu = 1.0
kp = 1.0/k_gain
kd = 0.05/k_gain
err_limit = 25 # print commanded velocity if above this
ctrl_mode = 0 # state of device (0 = rest, 1 = const_tau, 2 = zero-torque, 3 = static torq, 4 = cma torq)
gait_phase = 0 # 0 for stance, 1 for swing

# send or look for CAN message to link the exos
#can_exo = connectExos(leg, start_motor_mode, wait_time, exo_id)

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
print_int = int(0.5*rate+1) # divide rate by print int
initial_motor_flag = True
launch_ready = False
strike_avg = 0
launch_avg = 0
tstride_avg = 1.2
tstride = 0.0
tstance = 0.0
istride = 0
istance = 0
iswing = 0
Nstrides = 0
step_period = 1.2
heelstrike = False
prev_phase = 0.0

while(not exit_flag): ### main control and sensing loop
    cur_time = time.time()
    if (cur_time - start_time) >= dt: # wait for sample time
        start_time = cur_time # reset timer
        cnt += 1
        main_cnt += 1
        sensor_data = readSensors(adc, leg, pr, stance, ank_offset, thresh_strike) # get data from sensors
        #strike_avg, launch_avg = updateCycle(sensor_data[2], strike_avg, launch_avg, thresh_strike, thresh_launch)
        ### updating and filtering sensor data
        if cnt >= ts_len*rate: # filter and control once enough data to filter
            istride += 1
            if cnt%window==0:
                cnt = ts_len*rate
                torq_vec[:ts_len*rate] = torq_vec[-ts_len*rate:]
                pos_vec[:ts_len*rate] = pos_vec[-ts_len*rate:]
            pos_vec[cnt] = sensor_data[1]
            torq_vec[cnt] = sensor_data[0]
            pos_filt = sg.filtfilt(b2, a2, pos_vec[cnt-ts_len*rate:cnt], method='gust', irlen=100) # 134
            ankle_pos = pos_filt[-1]
            vel = (ankle_pos - pos_filt[-2])*rate
            torq_filt = sg.filtfilt(b1, a1, torq_vec[cnt-ts_len*rate:cnt], method='gust', irlen=100) # 69
            meas_torq = torq_filt[-1]

            if initial_motor_flag and ctrl_mode >= 3: # start init_time to not jump into cycle somewhere random
                init_time = cur_time
                initial_motor_flag = False
            last_step_time = (cur_time - init_time)%step_period
            phase, des_torq, des_future_torq = desiredTorque(tau_ramp, heelstrike_timing_adjust, torq_prof, step_period, last_step_time, init_tau, future_time) # determine desired torque
            if prev_phase > 70 and phase < 10.0:
                heelstrike = True
                istride = 0
            else:
                heelstrike = False
            prev_phase = phase

            if meas_torq < -5.0 and enable_motor: # error with torque measurements on heelspur
                print("Error with heelspur torque measurements.")
                enable_motor, initialized = killMotor(can0, m_id, stop_motor_mode)

            ### Motor control architecture
            if ctrl_mode == 0: # REST
                velmdes = 0
                des_torq = 0
                if cur_time - init_time >= 5:
                    ctrl_mode = 1
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
                if step_mode:
                    des_torq = init_tau*10
                    des_future_torq = init_tau*10
                else:
                    des_torq += init_tau
                    des_future_torq += init_tau
                pos_comp = (des_future_torq - meas_torq)*kt
                vel_comp = -mvel*kv
                il_comp = dataLRN[min(istride, max_cnt_lrn-D) + D, 0]
                velmdes = pos_comp + vel_comp + il_comp

            if ctrl_mode > 2: # iterative learning update
               dataLRN = updateLRN(istride, max_cnt_lrn, torq_filt, des_future_torq, dataLRN, mu, beta, KL)

            if enable_motor: # command torque and get motor data
                #if np.abs(velmdes) >= 25.6:
                #    print("clipping", velmdes)
                cmd_arr = [0.0, velmdes, 0.0, k_gain, 0.0]
                mpos, mvel, mtorq = motorUpdate(can0, m_id, cmd_arr, mpos, mvel, mtorq, wait_time)
        if plotter: ### send data to plotter / opt / save
            #sensor_data[2] = np.array([pos_comp, vel_comp, il_comp, velmdes/2.0])
            p.put([sensor_data, des_torq, mpos/drum_ratio, mvel/drum_ratio, mtorq, mot_temp, cpu_temp, ankle_pos, vel, meas_torq, phase, cur_time-init_time, ctrl_mode, enable_motor, velmdes,not gait_phase])
        if main_cnt%print_int == 0: # serial print control loop rate
            ctrl_rate = 1/((cur_time - int_time)/print_int)
            int_time = cur_time # print_int
            print("Ctrl loop:",np.round(ctrl_rate,1),"Hz   Ctrl mode:", ctrl_mode, "Meas torq (Nm):", np.round(meas_torq,2), "Des torq(Nm):", np.round(des_torq,2))
            if ctrl_rate < 0.7*rate and enable_motor: # error in control loop keeping up
                enable_motor, initialized = killMotor(can0, m_id, stop_motor_mode)
                print("Control loop rate error:", np.round(ctrl_rate,1))
                print(velmdes, mtorq, mpos, mvel, des_torq, gait_phase, np.round(tstride_avg,2), strike_avg, launch_avg)
        if main_cnt%button_mod == 0: # check exit and motor kill buttons
            if (mot_kill.value or can_button) and (cur_time - last_button_time > button_thresh):
                if prev_button:
                    last_button_time = cur_time
                    can_button = False
                    #if leg == 'LEFT':
                    #    can_exo.send(message(exo_id, stop_motor_mode))
                prev_button, enable_motor, ctrl_mode, initialized = stateMachine(ctrl_mode, enable_motor, prev_button, can0, m_id, stop_motor_mode, initialized)
            else:
                prev_button = False
            if leg == 'LEFT':
                exit_flag = button.value
                #if exit_flag:
                #    can_exo.send(message(exo_id, zero_position))
        if (cur_time - temp_time) >= period_temp: # temp sensor
            temp_time = cur_time
            mot_temp, cpu_temp = getTemp(adc, cpu, xp, yp)
        else:
            cpu_temp = 0.0

### EXITING
if enable_motor:
    killMotor(can0, m_id, stop_motor_mode)
disconnectExos()
if plotter: plot.terminate()
print("Program ending...")
time.sleep(1.0)
sys.exit(0)
