import numpy as np
import time
from mcp3008 import MCP3008
adc = MCP3008(bus=1,device=0)
#print("Applied voltage: %.2f" % (value / 1023.0 * 3.3) )

adc_vals = np.zeros(6)
samples = 1000
st = time.time()
for i in range(samples):
    res = 3.3/65536
    #adc_vals[0] = chan0.value*res
    adc_vals[0] = adc.read(0) # You can of course adapt the channel to be read out#adc.read([mcp3008.CH0])
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