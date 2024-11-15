import numpy as np
import time
weight = 68 # kg
height = 1.76 # meters
# TODO add load feature for other interp params
rise_scalar = 40.0
interp_params = np.array([[weight*0.64051748, 54.59, 0.58666667*rise_scalar, 9.98],[weight*0.72715882, 54.59, 0.70916667*rise_scalar, 9.98],[weight*0.80887392, 54.59, 0.85*rise_scalar, 9.98]])
max_torq = 55 # max exo can do is 55 Nm, make sure we don't try params above that
torq_bound = min(max_torq, weight)
# GA_params = np.array([[],[],[]])
static_params = np.array([40,54.8,33.4,9.1])#interp_params[1,:]#[41, 53.6, 24.1, 10.0]# #[46.1,54.8,33.4,9.1] -- pat DD parameters
speed_params = np.array([9.58239369128718,-24.2954102910252,16.2100069156388]) # for converting gait duration into m/s
torq_spds = np.array([0.0, 1.07, 1.33, 1.58, 1.8])

def interpParams(interp_params, torq_spds, cur_spd, torq_bound, inds=[0,2]):
    cur_spd = min(max(cur_spd, torq_spds[0]), torq_spds[-1])
    interp_temp = np.zeros((5,4))
    interp_temp[1:-1,:] = interp_params
    interp_temp[0,1:] = interp_params[0,1:] # set interpolation down to zero torque for slow speeds
    interp_temp[-1,1:] = interp_params[-1,1:]
    interp_temp[-1,0] = min(torq_bound, interp_params[2,0] + (interp_params[2,0]-interp_params[1,0])/(torq_spds[3]-torq_spds[2])*(torq_spds[4]-torq_spds[3]))
    static_params = interp_temp[0,:]
    for i, ind in enumerate(inds):
        static_params[ind] = np.interp(cur_spd, torq_spds, interp_temp[:,ind])
    return cur_spd, static_params
        
start_time = time.time()
iters = 1000
cur_spd_list = np.linspace(0, 1.7, iters).tolist()#[-5.0, 0.2, 1.0, 1.25, 1.42, 1.7]
for cur_spd in cur_spd_list:
    cur_spd, static_params = interpParams(interp_params, torq_spds, cur_spd, torq_bound)
    #print(cur_spd, torq_spds, static_params, interp_params)
print((time.time()-start_time)/iters)