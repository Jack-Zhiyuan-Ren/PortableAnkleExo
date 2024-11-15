import os
import time
#import busio
#import digitalio
#import board
#import adafruit_mcp3xxx.mcp3008 as MCP
import numpy as np
#from adafruit_mcp3xxx.analog_in import AnalogIn

#spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
#cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
#mcp = MCP.MCP3008(spi, cs)
from mcp3008 import MCP3008
adc = MCP3008(bus=1,device=0)

samples = 1000
for i in range(samples):
    
    chan0 = adc.read(4)#(mcp, MCP.P4)
    chan1 = adc.read(5)#AnalogIn(mcp, MCP.P5)
    chan2 = adc.read(6)#AnalogIn(mcp, MCP.P6)
    chan3 = adc.read(7)#AnalogIn(mcp, MCP.P7)
    res = 3.3/1023
    print("Pressure insoles (V):",np.round(chan0*res,1),np.round(chan1*res,1),np.round(chan2*res,1),np.round(chan3*res,1))
    time.sleep(0.1)