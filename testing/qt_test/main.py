import sys
from pyqtgraph.Qt import QtCore, QtGui
import threading
import numpy as np
import pyqtgraph as pg
import time
#from PlotData import Plot2D

class Plot2D(pg.GraphicsWindow):
    def __init__(self):
        pg.GraphicsWindow.__init__(self, title="Dibujar")
        self.traces = dict()
        self.resize(1000, 600)
        pg.setConfigOptions(antialias=True)
        #self.canvas = self.win.addPlot(title="Pytelemetry")
        self.waveform1 = self.addPlot(title='WAVEFORM1', row=1, col=1)
        self.waveform2 = self.addPlot(title='WAVEFORM2', row=2, col=1)
        
    @QtCore.pyqtSlot(tuple, tuple)
    def updateData(self, names, data):
        if names[0] in self.traces:
            for i, name in enumerate(self.traces):
                self.traces[name].setData(data[0], data[i+1])
                #self.traces[name2].setData(x, y2)
        else:
            name1, name2 = names
            x, y, y2 = data
            self.traces[name1] = self.waveform1.plot(x, y, pen='y', width=3)
            #elif name == "MPU":
            self.traces[name2] = self.waveform2.plot(x, y2, pen='y', width=3)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(tuple, tuple)

def create_data1(helper, name):
    t = np.arange(-3.0, 2.0, 0.01)
    i = 0.0
    while True:
        s = np.sin(2 * 2 * 3.1416 * t) / (2 * 3.1416 * t + i)
        s2 = np.sin(2 * 2 * 3.1416 * t) / (4 * 3.1416 * t + i)
        time.sleep(.001)
        helper.changedSignal.emit(name, (t, s, s2))
        i = i + 0.1

def create_data2(helper, name):
    t = np.arange(-3.0, 2.0, 0.01)
    i = 0.0
    while True:
        s = np.cos(2 * 2 * 3.1416 * t) / (2 * 3.1416 * t - i)
        time.sleep(.01)
        helper.changedSignal.emit(name, (t, s))
        i = i + 0.1

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    helper = Helper()
    plot = Plot2D()
    helper.changedSignal.connect(plot.updateData, QtCore.Qt.QueuedConnection)
    threading.Thread(target=create_data1, args=(helper, ("910D", "MPU")), daemon=True).start()
    #threading.Thread(target=create_data2, args=(helper, "MPU"), daemon=True).start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()