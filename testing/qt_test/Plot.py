from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from multiprocessing import Process, Manager, Queue
import time

def d(q):
    for i in range (10000):
        t = np.arange(0,3.0,0.001)
        s = np.sin(2 * np.pi * t + i*0.1)
        q.put([t,s])
        time.sleep(0.03)

def f(q):
    #app2 = QtGui.QApplication([])

    win2 = pg.GraphicsWindow(title="Basic plotting examples")
    win2.resize(1000,600)
    win2.setWindowTitle('pyqtgraph example: Plotting')
    p2 = win2.addPlot(title="Updating plot")
    curve = p2.plot(pen='y')

    def updateInProc(curve, q):
        #t = np.arange(0,3.0,0.01)
        #s = np.sin(2 * np.pi * t + updateInProc.i)
        
        if (q.qsize()>0):
            s = q.get()
            curve.setData(s)
        print(q.qsize())
            
        #updateInProc.i += 0.1

    #updateInProc.i = 0

    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: updateInProc(curve, q))
    timer.start(0)

    QtGui.QApplication.instance().exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    manager = Manager()
    data = manager.list()
    q = Queue()
    dat = Process(target=d, args=(q,))
    dat.start()
    p = Process(target=f, args=(q,))
    p.start()
    #input("Type any key to quit.")
    print("Waiting for graph window process to join...")
    #p.join()
    print("Process joined successfully. C YA !")