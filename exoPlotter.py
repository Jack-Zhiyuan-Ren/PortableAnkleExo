import numpy as np
import time
import sys
import threading
import numpy as np
import time
from multiprocessing import Process, Manager, Queue
from scipy import signal

plot = False
if plot:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    from pyqtgraph.dockarea import *
    class Plot2D(pg.GraphicsWindow):
        def __init__(self):
            pg.setConfigOption('background','w')
            pg.setConfigOption('foreground','k')
            pg.GraphicsWindow.__init__(self, title="Exo data stream")
            self.traces = dict()
            self.setGeometry(170,25,1000, 600)
            pg.setConfigOptions(antialias=True)
            #print(dir(self))
            #self.canvas = self.win.addPlot(title="Pytelemetry")
            self.waveform1 = self.addPlot(labels={'left': 'Torque (Nm)', 'bottom': 'Time (s)'}, row=1, col=1) #title='WAVEFORM1',
            self.waveform2 = self.addPlot(labels={'left': 'Voltage (V)', 'bottom': 'Time (s)'}, row=1, col=2) #title='WAVEFORM2',
            self.waveform3 = self.addPlot(labels={'left': 'Angle (rad)', 'bottom': 'Time (s)'}, row=2, col=1) #title='WAVEFORM2',
            self.waveform4 = self.addPlot(labels={'left': 'Vel (rad/s)', 'bottom': 'Time (s)'}, row=2, col=2) #title='WAVEFORM2',
            self.waveform5 = self.addPlot(labels={'left': 'Temp (C)', 'bottom': 'Time (s)'}, row=3, col=1) #title='WAVEFORM2',

            self.waveform1.setDownsampling(mode='peak')
            self.waveform2.setDownsampling(mode='peak')
            self.waveform3.setDownsampling(mode='peak')
            self.waveform4.setDownsampling(mode='peak')
            self.waveform5.setDownsampling(mode='peak')

            #self.waveform2.setYRange(-0.2, 3.5, padding=0)
            #self.waveform3.setYRange(-0.3, 0.6, padding=0)

            vb = self.addViewBox(row=3, col=2)
            colors = ['k', 'r', (255,128,0), 0.3, 'g', 'b', 'c', 'r', 'b', 'm', 'c', 'k', (255,102,102), (178,102,255)]
            name = ("Measured torque", "Desired torque", "Motor torque", "Heel", "Medial", "Toe", "Lateral", "Stance", "Motor angle", "Ankle angle", "Motor vel", "Ankle vel", "Motor temp", "CPU temp", 'CPU lim')
            y_offset = 0.4
            num_split = 7
            for i in range(len(colors)):
                t = pg.TextItem(name[i], color=colors[i], anchor = (0,0))
                if i < num_split:
                    t.setPos(0.0, 0.0-i*y_offset)
                else:
                    t.setPos(0.2, num_split*y_offset-i*y_offset)
                vb.addItem(t)

        @QtCore.pyqtSlot(tuple, tuple)
        def updateData(self, names, datat):
            ts, data, cpu_temp, mot_temp, ctrl_mode, enable_motor = datat
            if len(ts) > data.shape[1]:
                ts = ts[:data.shape[1]]
            ctrl_modes = ['Rest','Calibrating motor','Zero-torque','Static torque','CMA torque']
            if enable_motor:txt = 'Motor Enabled'
            else: txt = 'Motor Disabled'
            self.setWindowTitle('Mode:' + ctrl_modes[ctrl_mode] + '---' + txt)

            if names[0] in self.traces:
                for i, name in enumerate(self.traces):
                    if i < 12:
                        self.traces[name].setData(ts, data[i,:])
                    elif i == 12:
                        self.traces[name].setData(cpu_temp)
                    elif i == 13:
                        self.traces[name].setData(mot_temp)
                    elif i == 14:
                        self.traces[name].setData([0,len(cpu_temp)-1], [60,60])
            else:
                name1, name2, name3, name4, name5, name6, name7, name8, name9, name10, name11, name12, name13, name14, name15 = names
                colors = ['k', 'r', (255,128,0), 0.3, 'g', 'b', 'c', 'r', 'b', 'm', 'c', 'k', (255,102,102), (178,102,255)]
                self.traces[name1] = self.waveform1.plot(ts, data[0,:], pen=colors[0], width=3)
                self.traces[name2] = self.waveform1.plot(ts, data[1,:], pen=colors[1], width=3)
                self.traces[name3] = self.waveform1.plot(ts, data[2,:], pen=colors[2], width=3)
                # insole plot
                self.traces[name4] = self.waveform2.plot(ts, data[3,:], pen=colors[3], width=3)
                self.traces[name5] = self.waveform2.plot(ts, data[4,:], pen=colors[4], width=3)
                self.traces[name6] = self.waveform2.plot(ts, data[5,:], pen=colors[5], width=3)
                self.traces[name7] = self.waveform2.plot(ts, data[6,:], pen=colors[6], width=3)
                self.traces[name8] = self.waveform2.plot(ts, data[7,:], pen=colors[7], width=3)
                # motor/ankle pos
                self.traces[name9] = self.waveform3.plot(ts, data[8,:], pen=colors[8], width=3)
                self.traces[name10] = self.waveform3.plot(ts, data[9,:], pen=colors[9], width=3)
                # motor/ankle vel
                self.traces[name11] = self.waveform4.plot(ts, data[10,:], pen=colors[10], width=3)
                self.traces[name12] = self.waveform4.plot(ts, data[11,:], pen=colors[11], width=3)
                # temp
                self.traces[name13] = self.waveform5.plot(mot_temp, pen=colors[12], width=3)
                self.traces[name14] = self.waveform5.plot(cpu_temp, pen=colors[13], width=3)
                # add constant lines below here?
                self.traces[name15] = self.waveform5.plot([0,len(cpu_temp)-1], [60,60], pen=colors[12], style=QtCore.Qt.DashLine, width=3)

    class Helper(QtCore.QObject):
        changedSignal = QtCore.pyqtSignal(tuple, tuple)

    # main plotting function
    def plotting(saving, path, vis, p, c, rate, plot_rate, ts_len, window_s, cma):
        if vis:
            app = QtGui.QApplication([])
            helper = Helper()
            plot = Plot2D()
            helper.changedSignal.connect(plot.updateData, QtCore.Qt.QueuedConnection)
        else:
            helper = []

        tr = threading.Thread(target=createData, args=(saving, path, vis, helper, p, c, rate, plot_rate, ts_len, window_s, cma), daemon=True).start()
        if ((sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION')) and vis:
            QtGui.QApplication.instance().exec_()
        else:
            time.sleep(100000)

def createData(saving, path, vis, helper, p, c, rate, plot_rate, ts_len, window_s, cma):
    # setup initial plot
    print("Start plot thread...")
    plt_rem = rate//plot_rate
    num_bins = 30
    dcnt = 0
    window = rate*ts_len*window_s # TODO add window variable len
    data_mat = np.zeros((19, window))
    ts = np.arange(-ts_len, 0, 1./rate)
    name = ("Meas torq", "Des torq", "Mot torq", "Heel", "Medial", "Toe", "Lateral", "Stance", "Mot angle", "Ank angle", "Mot vel", "Ank vel", "Mot temp", "CPU temp", 'CPU lim')
    file_name = 'exodat_'
    file_cnt = 0
    # (0-2) act_torq, des_torq, mot_torq
    # (3-7) insoles, stance
    # (8-9) motor pos/vel
    # (10-11) ankle pos/vel
    cpu_list = []
    mot_list = []
    prev_strike_ind = 0
    cur_strike_ind = 0
    plot_finished = False
    while(not plot_finished): # loop waiting for new data to plot
        if(p.qsize()>0): # clearing the queues that may have old messages
            dcnt+=1
            if dcnt%window==0:
                dcnt = ts_len*rate
                if prev_strike_ind > 0:
                    prev_strike_ind -= rate*ts_len*(window_s-1)
                if saving:
                    np.save(path+file_name+str(file_cnt)+'.npy', data_mat[:,ts_len*rate:])
                    file_cnt += 1
                data_mat[:,:ts_len*rate] = data_mat[:,-ts_len*rate:]
                # pull new exo data and compute stuff
            sensor_data, des_torq, mpos, mvel, mtorq, mot_temp, cpu_temp, ankle_pos_filt, vel, torq_filt, phase, time, ctrl_mode, enable_motor, velmdes, gait_phase, il_comp, walk_spd = p.get()
            meas_torq, ankle_pos, pr, stance = sensor_data

            # make data mat bigger and only shift it like every 10 seconds (& keep prev window of data if want scrolling)
            data_mat[:3,dcnt] = [torq_filt, des_torq, mtorq] # [meas_torq, des_torq, mtorq]
            data_mat[3:7,dcnt] = pr
            data_mat[7:11,dcnt] = [gait_phase, mpos, ankle_pos_filt, mvel] # replace ankle_pos with mpos [stance, mpos, ankle_pos, mvel]
            data_mat[11:,dcnt] = [vel, phase, time, velmdes, mot_temp, cpu_temp, il_comp, walk_spd]

            if walk_spd != 0.0 and cma: # new step -- grab old data
                cur_strike_ind = dcnt - 4 # offset for heelstrike delay
                #print("formatting step data", cur_strike_ind, prev_strike_ind)
                if prev_strike_ind > 0:
                    ank_vec = data_mat[9,prev_strike_ind:cur_strike_ind]
                    ank_vel_vec = data_mat[11,prev_strike_ind:cur_strike_ind]
                    est_vec = np.zeros(2*num_bins)
                    est_vec[:num_bins] = signal.resample(ank_vec, num_bins, axis=0)# format ank ang/vel data
                    est_vec[num_bins:] = signal.resample(ank_vel_vec, num_bins, axis=0)# format ank ang/vel data
                    c.put([est_vec, walk_spd]) # passing binned data to cma thread
                prev_strike_ind = cur_strike_ind

            if cpu_temp != 0.0:
                cpu_list.append(cpu_temp)
                mot_list.append(mot_temp)
            
            if vis and dcnt%plt_rem==0:# update plot at defined interval
                if dcnt < ts_len*rate:
                    helper.changedSignal.emit(name, (ts[:dcnt], data_mat[:12,:dcnt], cpu_list, mot_list, ctrl_mode, enable_motor))
                else:
                    helper.changedSignal.emit(name, (ts, data_mat[:12,dcnt-ts_len*rate:dcnt], cpu_list, mot_list, ctrl_mode, enable_motor))
