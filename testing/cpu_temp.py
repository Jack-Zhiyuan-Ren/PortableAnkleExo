from gpiozero import CPUTemperature
import time
cpu = CPUTemperature()

ts = time.time()
for i in range(1000):
    temp = cpu.temperature # soft limit is 60C, stay below this
print((time.time()-ts)/1000.)