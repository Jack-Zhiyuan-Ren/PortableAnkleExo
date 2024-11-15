import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
mcp = MCP.MCP3008(spi, cs)

import numpy as np
x = 0.0 # measured resistance
yp = np.flip(np.array([15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150])) # resistances at temps
xp = np.flip(np.array([1.571,1.249,1,0.8057,0.6531,0.5327,0.4369,0.3603,0.2986,0.2488,0.2083,0.1752,0.1481,0.1258,0.1072,0.09177,0.07885,0.068,0.05886,0.05112,0.04454,0.03893,0.03417,0.03009,0.02654,0.02348,0.02083,0.01853])) # corresponding temps)

samples = 1000
for i in range(samples):
    
    chan0 = AnalogIn(mcp, MCP.P3)
    Rt = 0.1*(65536./chan0.value - 1.)
    temp = np.interp(Rt, xp, yp)
    print("Temp (C)", temp, Rt)
    time.sleep(0.1)