import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
#print(dir(board))
#print(dir(board.pin.GPIO.BCM))
#spi = busio.SPI(clock=board.SCK, MISO = board.MISO, MOSI = board.MOSI)
# D is the same as GPIO pin
spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
mcp = MCP.MCP3008(spi, cs)

chan0 = AnalogIn(mcp, MCP.P0)
chan1 = AnalogIn(mcp, MCP.P1)
chan2 = AnalogIn(mcp, MCP.P2)
chan3 = AnalogIn(mcp, MCP.P3)
chan4 = AnalogIn(mcp, MCP.P4)
chan5 = AnalogIn(mcp, MCP.P5)
import numpy as np
adc_vals = np.zeros(6)
samples = 1000
st = time.time()
for i in range(samples):
    #res = 3.3/65536
    chan0 = AnalogIn(mcp, MCP.P0)
    chan0.value#*res
    #adc_vals[1] = chan1.value*res
    #adc_vals[2] = chan2.value*res
    #adc_vals[3] = chan3.value*res
    #adc_vals[4] = chan4.value*res
    #adc_vals[5] = chan5.value*res
#     
tot_time = (time.time() - st)/samples
print(tot_time)
print("Samples per second:", 1./tot_time)


#print(chan0.value)
#time.sleep(0.1)