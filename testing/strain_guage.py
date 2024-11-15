import os
import time
import busio
import digitalio
import board
from mcp3008 import MCP3008

#import adafruit_mcp3xxx.mcp3008 as MCP
#from adafruit_mcp3xxx.analog_in import AnalogIn
import numpy as np
#spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
#cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
#mcp = MCP.MCP3008(spi, cs)
adc = MCP3008(bus=1,device=0)
samples = 1000
for i in range(samples):

    #chan0 = AnalogIn(mcp, MCP.P0)
    torque = 31.72*(adc.read(0)*3.3/1023.-0.4)+0.4
    force = (torque/0.117)/9.81
    print("Torque (Nm):",np.round(torque,1), "Force (kg):", np.round(force,1))
    time.sleep(0.1)
