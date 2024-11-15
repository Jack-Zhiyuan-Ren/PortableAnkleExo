import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def clear(q):
    try:
        while True:
            q.get_nowait()
    except:
        pass

# main plotting function
def plotting(p, rate, plot_rate, ts_len, torq_lim):
    # setup initial plot
    print("start plot thread")
    #plt.figure()
    #while(q.qsize()==0):# wait for first message
    #    pass
    #rate, ts_len, period_temp = q.get()
    dcnt = 0
    # setup data storage
    data_mat = np.zeros((12, rate*ts_len))
    while(True): # loop waiting for new data to plot
        if(p.qsize()>0): # clearing the queues that may have old messages
            upd_data = p.get()
            dcnt+=1
            
            if dcnt%rate==0:
                print(dcnt)
                #plt.plot(dcnt)
                #plt.show()
                # pull new exo data and compute stuff
                
            
            # update plot at defined interval
                          