import numpy as np

bout_data = np.load('../bout_data/bout_data_45_2.npy')
bout_dur, bout_break, bout_speed_list, bout_command_list = bout_data
for i in range(len(bout_dur)):
    print_str = str(bout_dur[i])+" "+bout_command_list[i]
    print(print_str)
