import os
import time
#import busio
#import digitalio
#import board
#import adafruit_mcp3xxx.mcp3008 as MCP
#from adafruit_mcp3xxx.analog_in import AnalogIn

#spi = busio.SPI(clock=board.D21, MISO = board.D19, MOSI = board.D20)
#cs = digitalio.DigitalInOut(board.D18)#board.D8)# D22
#mcp = MCP.MCP3008(spi, cs)
from mcp3008 import MCP3008
adc = MCP3008(bus=1,device=0)
samples = 1000
for i in range(samples):
    
    #chan0 = AnalogIn(mcp, MCP.P1)
    print("Ankle angle (rad):",(adc.read( channel = 1 )/1023.*3.14159))
    time.sleep(0.1)