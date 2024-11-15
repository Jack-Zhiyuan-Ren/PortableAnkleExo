import os
import can
from time import time, sleep
os.system('sudo ip link set can0 type can bitrate 1000000')
os.system('sudo ifconfig can0 up')

can0 = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')#'socketcan_ctypes')

def message(arb_id, data):
    return can.Message(arbitration_id=arb_id, data=data, extended_id = False)

# constants for converting motor commands
PMIN = -12.5#95.5#12.5#-95.5
PMAX = -PMIN
VMIN = -25.64#6.57#-30.
VMAX = -VMIN
TMIN = -18#54.#-18.
TMAX = -TMIN
KPMIN = 0.
KPMAX = 500.
KDMIN = 0.
KDMAX = 5.
TEST_POS = 0.

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
    #print(data[0],data[1], data)
    id_val = data[0]
    p_int = (data[1]<<8)|data[2]
    v_int = (data[3]<<4)|(data[4]>>4)
    t_int = ((data[4]&0xF)<<8)|data[5]
    # convert to floats
    p = uint_to_float(p_int, PMIN, PMAX, 16)
    v = uint_to_float(v_int, VMIN, VMAX, 12)
    t = uint_to_float(t_int, TMIN, TMAX, 12)
    if id_val != 1:
        print("ID val:",id_val) # check if id = 1?
    return p,v,t

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

def pack_mcmd(p_des, v_des, t_ff):
    # convert floats to ints
    p_int = float_to_uint(p_des, PMIN, PMAX, 16)
    v_int = float_to_uint(v_des, VMIN, VMAX, 12)
    t_int = float_to_uint(t_ff, TMIN, TMAX, 12)
    # pack ints into buffer message
    msg = []
    msg.append(0x01)
    msg.append(p_int>>8)
    msg.append(p_int&0xFF)
    msg.append(v_int>>4)
    msg.append((((v_int&0xF)<<4)) | (t_int>>8))
    msg.append(t_int&0xFF)
    return msg

# testing functions
# print(float_to_uint(0., -1., 1., 12))
# print(uint_to_float(2047, -1., 1., 12))
#msg = pack_cmd(5., 0., 5., 0., 0.)
#print(msg)
#msg = pack_cmd(0,0,0,0,0)
#print(msg)
#msg = pack_cmd(0,7,0,2,0)
#print(msg)
#msg = pack_cmd(0,0,0,0,5)
#print(msg)
# mmsg = pack_mcmd(1.0, 2.0, 3.0)
# output = unpack_msg(mmsg)
# print(output)

start_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFC]
stop_motor_mode = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFD]
zero_position = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE]
m_id = 0x01 # motor ID #0x123
wait_time = 0.001
can0.send(message(m_id, start_motor_mode)) # zero motor pos
new_msg = can0.recv(wait_time) # time is timeout waiting for message
can0.send(message(m_id, zero_position))
#try:
#    print("ERR:",new_msg.error_state_indicator, new_msg.is_error_frame)
#    print(new_msg)
#    print('P,V,T',unpack_msg(new_msg.data))
#except: pass
#print("Sending start command")
#can0.send(message(m_id, start_motor_mode)) # enabling motor mode
#can0.send(message(m_id, zero_position)) # zero motor pos
sleep(1)
new_msg = can0.recv(wait_time) # time is timeout waiting for message
P = 0.0
V = 0.0
T = 0.0
try:
    print(new_msg)
    #print(new_msg.arbitration_id, new_msg.channel, new_msg.bitrate_switch, dir(new_msg))
    P,V,T = readMotorMsg(new_msg, P, V, T)#unpack_msg(new_msg.data)
    print('P,V,T',P,V,T)
except: pass
#print(new_msg.data)#, new_msg.timestamp)

# send a change of position
#cmd = pack_cmd(p_des, v_des, kp, kd, t_ff)
#cmd = pack_cmd(1.0, 0.0, 1.0, 0.0, 0.0)
cmd = pack_cmd(0.0, 0.1, 0.0, 0.1, 0.0)
print(cmd)
#can0.send(message(m_id, cmd)) # enabling motor mode
for i in range(6):
    can0.send(message(m_id, cmd))
    new_msg = can0.recv(wait_time)
    P,V,T = readMotorMsg(new_msg, P, V, T)
    print(P,V,T)
    sleep(0.1)

st = time()
num_msgs = 1000
for i in range(num_msgs):
    can0.send(message(m_id, cmd))
    new_msg = can0.recv(wait_time)
    readMotorMsg(new_msg, P, V, T)

tot_time = time() - st
print("Time per msg:", tot_time/num_msgs)
#new_msg = can0.recv(wait_time) # time is timeout waiting for message
#try:
   #print(new_msg)
   #print('P,V,T',unpack_msg(new_msg.data))
#except: pass
#shutting down

sleep(1)
print("Shutting down can")
can0.send(message(m_id, stop_motor_mode)) # disable motor mode
os.system('sudo ifconfig can0 down')
