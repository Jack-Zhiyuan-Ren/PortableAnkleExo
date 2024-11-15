import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import can

spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
mcp = MCP.MCP3008(spi, cs)
chan0 = AnalogIn(mcp, MCP.P0)
print(chan0.value)



os.system('sudo ip link set can0 type can bitrate 1000000')
os.system('sudo ifconfig can0 up')
os.system('sudo ip link set can1 type can bitrate 1000000')
os.system('sudo ifconfig can1 up')

can0 = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')
can1 = can.interface.Bus(channel = 'can1', bustype = 'socketcan_ctypes')

msg = can.Message(arbitration_id=0x123, data=[0,1,2,3,4,5,6,7], extended_id = False)
can0.send(msg)
new_msg = can1.recv(1.0)
print(new_msg)
#new_msg = can1.recv(1.0) # time is timeout waiting for message
print('Can1:', new_msg)
new_msg = can0.recv(1.0)
try:
    print(new_msg)
    print(dir(new_msg), len(new_msg))

    msg_time = new_msg.timestamp # float
    msg_data = new_msg.data # bytearray
    print(msg_time, msg_data)
except:
    pass



chan0 = AnalogIn(mcp, MCP.P0)
print(chan0.value)

os.system('sudo ifconfig can0 down')
os.system('sudo ifconfig can1 down')
