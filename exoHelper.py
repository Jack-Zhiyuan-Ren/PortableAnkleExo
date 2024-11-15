import numpy as np
import can
from chspy import CubicHermiteSpline
import os
from natsort import natsorted

# constants for motor stuff
PMIN = -12.5#95.5
PMAX = -PMIN
VMIN = -25.64
VMAX = -VMIN
TMIN = -18#54
TMAX = -TMIN
KPMIN = 0.
KPMAX = 500.
KDMIN = 0.
KDMAX = 5.

PMIN2 = 0.0 # for pk torq
PMAX2 = 60.0
VMIN2 = 10.0 # for rise time
VMAX2 = 40.0
TMIN2 = 0. # rest are for cur_bin
TMAX2 =  5.
KPMIN2 = 0.
KPMAX2 = 5.
KDMIN2 = 0.
KDMAX2 = 5.

def loadParams(new_param, norm_param, weight, rise_scalar):
    new_param[0] = norm_param[0]*weight
    new_param[2] = norm_param[1]*rise_scalar
    return new_param

def grabMeans(save_mean_data, gen_counter):
    spd_bins = save_mean_data.shape[0]
    gen_tot = np.sum(gen_counter)
    save = []
    for i in range(spd_bins):
        save.append(np.expand_dims(save_mean_data[i,gen_tot,:],axis=0))
    savecat = np.concatenate(save,axis=0)
    return savecat

# def play_audio(fname, audio, wait=False):
#     if audio:
#         path = os.getcwd() + '/sound_files/'
#         try:
#             p = vlc.MediaPlayer(path+fname+'.mp3')
#             p.play()
#             #print("Playing:", fname)
#             #print(path, fname, '.mp3')
#             if wait:
#                 time.sleep(0.5)
#                 while p.get_state() == vlc.State.Playing:
#                     pass
#         except:
#             print("Error playing:", fname)

def interpParams(interp_params, torq_spds, cur_spd, torq_bound, inds=[0,2]):
    interp_temp = np.zeros((5,4))
    interp_temp[1:-1,:] = interp_params
    interp_temp[0,1:] = interp_params[0,1:] # set interpolation down to zero torque for slow speeds
    interp_temp[-1,1:] = interp_params[-1,1:]
    interp_temp[-1,0] = min(torq_bound, interp_params[2,0] + (interp_params[2,0]-interp_params[1,0])/(torq_spds[3]-torq_spds[2])*(torq_spds[4]-torq_spds[3]))
    static_params = interp_temp[0,:]
    for i, ind in enumerate(inds):
        static_params[ind] = np.interp(cur_spd, torq_spds, interp_temp[:,ind])
    return static_params

def loadOffline(save_dir):
    files = natsorted(os.listdir(save_dir))
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
    return data

def connectExos(leg, start_motor_mode, wait_time, exo_id = 0x25):
    os.system('sudo ip link set can1 type can bitrate 1000000')
    os.system('sudo ifconfig can1 up')
    can_exo = can.interface.Bus(channel = 'can1', bustype = 'socketcan_ctypes')#'socketcan_ctypes')    print("Connecting to motor")
    if leg == 'LEFT':
        print("Sending start command to other exo.")
        can_exo.send(message(exo_id, start_motor_mode)) # zero motor pos
    elif leg == 'RIGHT':
        new_msg = None
        print("Waiting for message from other exo...")
        while(new_msg != start_motor_mode):
            new_msg_dat = can_exo.recv(10000)
            try:
                new_msg = list(new_msg_dat.data)
            except:
                pass
        print("Message received from other exo!")

    return can_exo

def disconnectExos():
    os.system('sudo ifconfig can1 down')

# Initialize CAN
def connectMotor(m_id, start_motor_mode, wait_time, zero_position, mpos, mvel, mtorq):
    os.system('sudo ip link set can0 type can bitrate 1000000')
    os.system('sudo ifconfig can0 up')
    can0 = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')#'socketcan_ctypes')    print("Connecting to motor")
    can0.send(message(m_id, start_motor_mode)) # zero motor pos
    _ = can0.recv(wait_time) # time is timeout waiting for message
    mpos, mvel, mtorq = zeroMotor(can0, m_id, zero_position, mpos, mvel, mtorq, wait_time)
    return True, True, can0, mpos, mvel, mtorq

def updateCycle(pr, strike_avg, launch_avg, thresh_strike, thresh_launch):
    """ The function updates the 'strike_avg' and 'launch_avg'
        based on the current pressure sensor values and threshold values
        pressure sensors are [heel, medial, toe, lateral]

        @params
        pr: a list containing pressure sensor values
        strike_avg: an integer representing the current number of consecutive cycles where the pressure values indicate a foot strike
        launch_avg: an integer representing the current number of consecutive cycles where the pressure values indicate a foot launch
        thresh_strike: an integer representing the threshold pressure value for a foot strike
        thresh_launch: an integer representing the threshold pressure value for a foot launch
    """

    if (pr[0] <= thresh_strike or pr[1] <= thresh_strike or pr[2] <= thresh_strike):
        strike_avg += 1
    else:
        strike_avg = 0 # reset

    if (pr[2] >= thresh_launch) and (pr[0] >= thresh_strike): #ps[1] >= thresh_launch
        launch_avg += 1
    else:
        launch_avg = 0
    return strike_avg, launch_avg#, prev_strike, prev_launch

def checkToeOff(pr, thresh_mid, launch_ready, launch_avg, num_launch):
    check5 = False
    if pr[1] <= thresh_mid and pr[2] <= thresh_mid:# and pr[3] <= thresh_mid:
        launch_ready = True
    if launch_ready and launch_avg >= num_launch:
        check5 = True
    return check5, launch_ready

def zeroMotor(can0, m_id, zero_position, mpos, mvel, mtorq, wait_time):
    can0.send(message(m_id, zero_position))
    new_msg = can0.recv(wait_time) # time is timeout waiting for message
    mpos, mvel, mtorq = readMotorMsg(new_msg, mpos, mvel, mtorq)
    return mpos, mvel, mtorq

def killMotor(can0, m_id, stop_motor_mode):
    can0.send(message(m_id, stop_motor_mode))
    os.system('sudo ifconfig can0 down')
    return False, False # set enable_motor to False

def stateMachine(d, ctrl_mode, enable_motor, prev_button, can0, m_id, stop_motor_mode, initialized):
    if prev_button:
        if ctrl_mode == 0: # transition from rest to calibrate motor cable
            ctrl_mode += 1
            #enable_motor = initialized
        elif ctrl_mode == 1 or ctrl_mode == 3: # kill motor
            if enable_motor:
                print("Killing motor...")
                enable_motor, initialized = killMotor(can0, m_id, stop_motor_mode)
                ctrl_mode = 0
                d.put([[0,0,0]]) # send signal to CMA to pause
            else:
                if ctrl_mode == 1:
                    ctrl_mode += 1
                else:
                    ctrl_mode = 0
                    d.put([[0,0,0]]) # send signal to CMA to pause
            #initialized = False
        elif ctrl_mode == 2: # switch to torque control from ZT
            ctrl_mode += 1
            d.put([[1,1,1]]) # send signal to CMA to start

        elif ctrl_mode == 4:
            ctrl_mode = 0
        prev_button = False

    else:
        prev_button = True

    return prev_button, enable_motor, ctrl_mode, initialized

def updateLRN(istride, max_cnt_lrn, torq_filt, des_future_torq, dataLRN, mu, beta, KL):
    cnt_lrn = min(istride, max_cnt_lrn)
    # update future err
    etau = torq_filt[-1] - des_future_torq
    dposmLRN = dataLRN[cnt_lrn, 0]
    etaufilt = (1-mu)*dataLRN[cnt_lrn,1] + mu*etau
    dataLRN[cnt_lrn, 0] = beta*dposmLRN - KL*etaufilt
    dataLRN[cnt_lrn, 1] = etaufilt
    return dataLRN

def desiredTorque(tau_ramp, heelstrike_timing_adjust, torq_prof, tstride_avg, tstride, init_tau, dt):
    """ Calculate the current gait phase (%) with respect to the gait cycle
        Get the corresponding torque value from the torque profile
        return gait phase, desired torque value, and desired future torque value

        @params
        tau_ramp:    a scaling factor applied to the desired torque profile [JWQ: why need to amplify the torque?]
        heelstrike_timing_adjust: a timing adjustment value to shift the timing of the torque profile in time with the heel strike.
        torq_prof:   a torque profile object that contains a series of torque values for different phases of the gait cycle.
        tstride_avg: the average duration of a gait cycle.
        tstride:     the current time within a gait cycle.
        init_tau:    the initial torque value. [not used in this function]
        dt:          the time step used to calculate the future desired torque.
    """
    phase = (tstride + heelstrike_timing_adjust)/tstride_avg * 100  # calculate the current phase of the gait cycle (gait cycle percentage)
    des_torq = torq_prof.get_state(phase)[0] * tau_ramp             # tau for current, get the corresponding torque value from torque profile based on gait phase

    if dt != 0.0:   # if dt is non-zero, it calculates the phase value for a future time step and gets the corresponding torque value
        phase2 = (tstride + dt + heelstrike_timing_adjust)/tstride_avg * 100 # phase for desired motor torque in the future
        des_future_torq = torq_prof.get_state(phase2)[0] * tau_ramp
    else:
        des_future_torq = des_torq

    return phase, des_torq, des_future_torq

def swingvel(posa,vela,posm,spool_ratio,k_zt,dtt,slack,offset, phase=0.0, phase_cutoff=105.0, end_phase = 95.0, phase_thresh=105.0):
    if phase > phase_cutoff and phase < phase_thresh:
        slack_temp = slack*(1-(phase-phase_cutoff)/(end_phase - phase_cutoff))
        slack = max(-slack*0.5, slack_temp)
        #print(slack, phase)

    posmdes = (posa + vela * dtt) * (spool_ratio) - offset - slack # Change in motor position due to ankle angle, motor offset, and slack
    velmdes = (posmdes - posm)*k_zt # Desired motor velocity
    return velmdes

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

def motorUpdate(can0, m_id, cmd_arr, mpos, mvel, mtorq, wait_time):
    # cmd array: p_des, v_des, kp, kd, t_ff
    cmd = pack_cmd(cmd_arr[0],cmd_arr[1],cmd_arr[2],cmd_arr[3],cmd_arr[4])
    can0.send(message(m_id, cmd))
    new_msg = can0.recv(wait_time)
    mpos, mvel, mtorq = readMotorMsg(new_msg, mpos, mvel, mtorq)
    return mpos, mvel, mtorq

def exoUnpack(interp_params, msg):
    try:
        pk_torq, rise_time, cur_bin = unpack_msg2(msg.data)
        cur_bin = int(cur_bin)
        interp_params[cur_bin, 0] = pk_torq
        interp_params[cur_bin, 2] = rise_time
    except:
        print("Error unpacking exo parameter message:", unpack_msg2(msg))
        cur_bin = -1
    return interp_params, cur_bin

def exoUpdate(can0, m_id, pk_torq, rise_time, cur_bin, wait_time):
    # cmd array: p_des, v_des, kp, kd, t_ff
    cmd = pack_mcmd(pk_torq, rise_time, cur_bin)    # packs the desired torque command into a byte array to be sent over the CAN bus
    can0.send(message(m_id, cmd))                   # create a new CAN message with m_id motor and the command byte array 'cmd', sends the CAN message over the bus to m_id motor
    #new_msg = can0.recv(wait_time)
    #mpos, mvel, mtorq = readMotorMsg(new_msg, mpos, mvel, mtorq)
    #return mpos, mvel, mtorq

def getTemp(adc, cpu, xp, yp):
    Rt = 0.1*(1023./(adc.read(3)+0.0001) - 1.)
    mot_temp = np.round(np.interp(Rt, xp, yp),1)
    cpu_temp = np.round(cpu.temperature,1)
    #print('Mot (C):', mot_temp, 'CPU (C):', cpu_temp)
    return mot_temp, cpu_temp

def float_to_uint(x, xmin, xmax, bits):
    span = xmax-xmin
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax
    convert = int((x - xmin)*(((1<<bits)-1)/span))
    return convert

def uint_to_float(x, xmin, xmax, bits):
    span = xmax-xmin
    int_val = float(x)*span/(float((1<<bits)-1))+xmin
    return int_val

def unpack_msg(data):
    if data == None:
        return None
    #else:
    #    print(data[0],data[1], data)
    id_val = data[0]
    p_int = (data[1]<<8)|data[2]
    v_int = (data[3]<<4)|(data[4]>>4)
    t_int = ((data[4]&0xF)<<8)|data[5]
    # convert to floats
    p = uint_to_float(p_int, PMIN, PMAX, 16)
    v = uint_to_float(v_int, VMIN, VMAX, 12)
    t = uint_to_float(t_int, TMIN, TMAX, 12)
    if id_val != 1:
        print("ID val:",id_val, len(data)) # check if id = 1?
    return p,v,t # position, velocity, torque

def unpack_msg2(data):
    if data == None:
        return None
    #else:
    #    print(data[0],data[1], data)
    id_val = data[0]
    p_int = (data[1]<<8)|data[2]
    v_int = (data[3]<<4)|(data[4]>>4)
    t_int = ((data[4]&0xF)<<8)|data[5]
    # convert to floats
    p = uint_to_float(p_int, PMIN2, PMAX2, 16)
    v = uint_to_float(v_int, VMIN2, VMAX2, 12)
    t = uint_to_float(t_int, TMIN2, TMAX2, 12)
    if id_val != 1:
        print("ID val:",id_val, len(data)) # check if id = 1?
    return p,v,t # position, velocity, torque


def pack_mcmd(p_des, v_des, t_ff):
    # convert floats to ints
    p_int = float_to_uint(p_des, PMIN2, PMAX2, 16)
    v_int = float_to_uint(v_des, VMIN2, VMAX2, 12)
    t_int = float_to_uint(t_ff, TMIN2, TMAX2, 12)
    # pack ints into buffer message
    msg = []
    msg.append(0x01)
    msg.append(p_int>>8)
    msg.append(p_int&0xFF)
    msg.append(v_int>>4)
    msg.append((((v_int&0xF)<<4)) | (t_int>>8))
    msg.append(t_int&0xFF)
    return msg

def pack_cmd2(p_des, v_des, kp, kd, t_ff):
    # convert floats to ints
    p_int = float_to_uint(p_des, PMIN2, PMAX2, 16)
    v_int = float_to_uint(v_des, VMIN2, VMAX2, 12)
    kp_int = float_to_uint(kp, KPMIN2, KPMAX2, 12)
    kd_int = float_to_uint(kd, KDMIN2, KDMAX2, 12)
    t_int = float_to_uint(t_ff, TMIN2, TMAX2, 12)
    # pack ints into buffer message
    msg = []
    msg.append(p_int>>8)
    msg.append(p_int&0xFF)
    msg.append(v_int>>4)
    msg.append((((v_int&0xF)<<4)) | (kp_int>>8))
    msg.append(kp_int&0xFF)
    msg.append(kd_int>>4)
    msg.append((((kd_int&0xF)<<4)) | (t_int>>8))
    msg.append(t_int&0xFF)
    return msg

def pack_cmd(p_des, v_des, kp, kd, t_ff):
    # convert floats to ints
    p_int = float_to_uint(p_des, PMIN, PMAX, 16)
    v_int = float_to_uint(v_des, VMIN, VMAX, 12)
    kp_int = float_to_uint(kp, KPMIN, KPMAX, 12)
    kd_int = float_to_uint(kd, KDMIN, KDMAX, 12)
    t_int = float_to_uint(t_ff, TMIN, TMAX, 12)
    # pack ints into buffer message
    msg = []
    msg.append(p_int>>8)
    msg.append(p_int&0xFF)
    msg.append(v_int>>4)
    msg.append((((v_int&0xF)<<4)) | (kp_int>>8))
    msg.append(kp_int&0xFF)
    msg.append(kd_int>>4)
    msg.append((((kd_int&0xF)<<4)) | (t_int>>8))
    msg.append(t_int&0xFF)
    return msg

def readMotorMsg(new_msg, P, V, T):
    try:
        #print(new_msg)
        #print(new_msg.arbitration_id, new_msg.channel, new_msg.bitrate_switch, dir(new_msg))
        P,V,T = unpack_msg(new_msg.data)
        #print('P,V,T',unpack_msg(new_msg.data))
        return P,V,T
    except:
        print("Motor message error...")
        return P,V,T

def message(arb_id, data):
    return can.Message(arbitration_id=arb_id, data=data, extended_id = False)

def initEncoder(adc):
    " Initialize and returns the offset value of an ankle encoder "

    ank_offset = adc.read(1)/1023.*3.14159
    return ank_offset

def readSensors(adc, leg, pr, stance, ank_offset, pressure_thresh = 0.7):
    """ Read the measured torque (ADC pin 0), ankle position (ADC pin 1),
        The torque needs to be calculated using a calibration formula
        The readout ankle position from ankle encoder needs to be converted from voltage to radians,
        and substract an offset to get the actual ankle position. The left and right side are opposite value
        Read pressure sensor (ADC pin 4-7), and binary stance phase

        return the measured torque. ankle position, pressure, stance phase flag
    """

    res = 3.3/1023

    if leg == 'LEFT':
        meas_torq = 31.72*(adc.read(0)*res-0.4)+0.4         # Calf cuff torq measurement (Nm)
        ankle_pos = adc.read(1)/1023.*3.14159 - ank_offset  #*3.3/3.3 # Ankle encoder position (rad)
    else:
        meas_torq = 29.3/1.11*(adc.read(0)*res-0.4)         # 0.91*28.3
        ankle_pos = ank_offset - adc.read(1)/1023.*3.14159  #*3.3/3.3 # Ankle encoder position (rad)

    # read pressure sensor
    pr[0] = adc.read(4)*res
    pr[1] = adc.read(5)*res
    pr[2] = adc.read(6)*res
    pr[3] = adc.read(7)*res

    # Get pr[2] and pr[3] equal the minimum value of pr[2] and pr[3]
    if leg == 'RIGHT':
        if pr[2] < pr[3]:
            pr[3] = pr[2]
        else:
            pr[2] = pr[3]
    if leg == 'LEFT':
        if pr[2] < pr[3]:
            pr[3] = pr[2]
        else:
            pr[2] = pr[3]

    # Use pressure sensor to determine the stance phase
    if np.min(pr) <= pressure_thresh:
        stance = True
    else:
        stance = False
    # update stance based on these values
    # filter here or later?
    return [meas_torq, ankle_pos, pr, stance]
