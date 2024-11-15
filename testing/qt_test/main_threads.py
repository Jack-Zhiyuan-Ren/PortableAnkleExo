import sys
from pyqtgraph.Qt import QtCore, QtGui
import threading
import numpy as np
import pyqtgraph as pg
import time
from multiprocessing import Process, Manager, Queue

class Plot2D(pg.GraphicsWindow):
    def __init__(self):
        pg.GraphicsWindow.__init__(self, title="Dibujar")
        self.traces = dict()
        self.resize(1000, 600)
        pg.setConfigOptions(antialias=True)
        #self.canvas = self.win.addPlot(title="Pytelemetry")
        self.waveform1 = self.addPlot(title='WAVEFORM1', row=1, col=1)
        self.waveform1.setDownsampling(mode='peak')
        self.waveform2 = self.addPlot(title='WAVEFORM2', row=2, col=1)
        
    @QtCore.pyqtSlot(tuple, np.ndarray)
    def updateData(self, names, datat):
        data = datat
        if names[0] in self.traces:
            for i, name in enumerate(self.traces):
                self.traces[name].setData(data[0,:], data[i+1,:])
        else:
            name1, name2 = names
            #x, y, y2 = data
            self.traces[name1] = self.waveform1.plot(data[0,:], data[1,:], pen='y', width=3)
            #elif name == "MPU":
            self.traces[name2] = self.waveform2.plot(data[0,:], data[2,:], pen='y', width=3)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(tuple, np.ndarray)

def create_data(vis, q, helper, name):
    dcnt = 0
    plt_len = 700
    window = plt_len*3
    downsample = 5
    plot_data = np.zeros((3,window))
    while True:
        if (q.qsize()>0):
            t,s,s2 = q.get()
            plot_data[:,dcnt] = [t,s,s2]
            dcnt += 1
            if dcnt%window == 0:
                print("reset", q.qsize())
                dcnt = plt_len
                plot_data[:,:plt_len] = plot_data[:,-plt_len:]
            #print(dcnt)
            #time.sleep(.001)
            #print(plot_data[0,:10], plot_data[1,:10])
            if vis and dcnt%downsample == 0:
                if dcnt < plt_len:
                    helper.changedSignal.emit(name, (plot_data[:,:dcnt]))
                else:
                    helper.changedSignal.emit(name, (plot_data[:,dcnt-plt_len:dcnt]))

def plotter(vis, q, args, names):
    if vis:
        app = QtGui.QApplication(args)
        helper = Helper()
        plot = Plot2D()
        helper.changedSignal.connect(plot.updateData, QtCore.Qt.QueuedConnection)
    else:
        helper = []
    tr = threading.Thread(target=create_data, args=(vis, q, helper, names), daemon=False).start()
    if ((sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION')) and vis:
        QtGui.QApplication.instance().exec_()
        tr.terminate()
    else:
        time.sleep(100000)
        
if __name__ == '__main__':
    vis = True
    q = Queue()

    plt = Process(target=plotter, args=(vis, q, sys.argv,("910D", "MPU")), daemon=True)
    plt.start()
    t = 0.01
    tstep = 0.02
    while(t < 15):
        s = np.sin(2 * 2 * 3.1416 * t) / (2 * 3.1416 * t)
        s2 = np.sin(2 * 2 * 3.1416 * t) / (4 * 3.1416 * t)
        q.put([t,s,s2])
        t += tstep
        time.sleep(tstep)
    plt.terminate()
    print("done")
    
    
    
