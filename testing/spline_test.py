from chspy import CubicHermiteSpline
#import chspy
import matplotlib.pyplot as plt; plt.rcdefaults()


def torque_profile(params):
    spline = CubicHermiteSpline(n=1)
    spline.add((0, [0], [0])) # init pt
    spline.add((params[1]-params[2], [0], [0])) # start onset
    spline.add((params[1], [params[0]], [0])) # pk torque
    spline.add((params[1]+params[3], [0], [0])) # end torque
    spline.add((100, [0], [0])) # last_pt
    return spline

params = [46.1,54.8,33.4,9.1]

spline_list = []
fig,ax = plt.subplots()#figsize = (plt_double_width/2.5, plt_height*0.8))
spline = torque_profile(params)
spline_list.append(spline)
spline.plot(ax, markersize=0, c='b')
ax.set_xlabel("Gait Cycle (%)")
ax.set_ylabel("Torque Profile (Nm/kg)")
#plt.show()
import time
nums = 1000
ts = time.time()
for i in range(nums):
    state = spline.get_state(i%55)
#print(spline.get_state(55))
print((time.time()-ts)/nums)

#print(spline, dir(spline))