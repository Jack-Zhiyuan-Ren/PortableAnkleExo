from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import sys

class Plot2D(pg.GraphicsWindow):
    def __init__(self):
        pg.GraphicsWindow.__init__(self, title="Dibujar")
        self.traces = dict()
        self.resize(1000, 600)
        pg.setConfigOptions(antialias=True)
        #self.canvas = self.win.addPlot(title="Pytelemetry")
        self.waveform1 = self.addPlot(title='WAVEFORM1', row=1, col=1)
        self.waveform2 = self.addPlot(title='WAVEFORM2', row=2, col=1)

    def set_plotdata(self, name, x, y):
        if name in self.traces:
            self.traces[name].setData(x, y)
        else:
            if name == "910D":
                self.traces[name] = self.waveform1.plot(x, y, pen='y', width=3)
            elif name == "MPU":
                self.traces[name] = self.waveform2.plot(x, y, pen='y', width=3)

    @QtCore.pyqtSlot(str, tuple)
    def updateData(self, name, ptm):
        x, y = ptm
        self.set_plotdata(name, x, y)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    plot = Plot2D()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()