
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import pynvml

def reset_axes(ax,minx,maxx,miny,maxy):
    plt.sca(ax)
    plt.cla()
    ax.set_xlim([minx, maxx])
    ax.set_ylim([miny, maxy])
    ax.set_axis_bgcolor('black')

def gpuInfoList():
    pynvml.nvmlInit()
    num_gpu = pynvml.nvmlDeviceGetCount()
    info = []
    for i in range(0, num_gpu):
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        memory= pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        mem_util = (memory.total -memory.free )*100/float(memory.total)

        util_rate = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        device_name = pynvml.nvmlDeviceGetName(gpu_handle)
        info.append((i, device_name, util_rate.gpu,mem_util))
    return info


utilization_info = []
memory_info=[]

ion()
fig, (ax_util,ax_mem) = plt.subplots(1,2)


reset_axes(ax_util,0,100,0,100)
reset_axes(ax_mem,0,100,0,100)
plt.show()

max_data_lim=20 # max number of observation on x axis (same for both plots)

while True:
    try:
        if len(utilization_info) != max_data_lim:

            dt = datetime.datetime.now()
            util = gpuInfoList()

            utilization_info.append([dt] + [x[2] for x in util])
            memory_info.append([dt]+[x[3] for x in util])
            continue
        else:

            utilization_info.pop(0)
            memory_info.pop(0)
            dt = datetime.datetime.now()
            util = gpuInfoList()

            utilization_info.append([dt] + [x[2] for x in util])
            memory_info.append([dt] + [x[3] for x in util])

            reset_axes(ax_util, 0, 100, 0, 100)
            reset_axes(ax_mem, 0, 100, 0, 100)

            util_frame = pd.DataFrame(utilization_info,columns=['Time']+['GPU %i - %s' % (x[0], x[1]) for x in util]).set_index(['Time'])
            util_frame.plot(ax=ax_util, yticks=np.linspace(0,100,21))
            patches_util, labels_util = ax_util.get_legend_handles_labels()
            ax_util.legend(patches_util, labels_util, loc='upper right',bbox_to_anchor=(1, 1.1))

            mem_frame = pd.DataFrame(memory_info,columns=['Time']+['GPU %i - %s' % (x[0], x[1]) for x in util]).set_index(['Time'])
            mem_frame.plot(ax=ax_mem,yticks=np.linspace(0,100,21))
            patches_mem, labels_mem = ax_mem.get_legend_handles_labels()
            ax_mem.legend(patches_mem, labels_mem, loc='upper right', bbox_to_anchor=(1, 1.1))

            vals_util = ax_util.get_yticks()
            vals_mem = ax_mem.get_yticks()

            ax_util.set_yticklabels(['{:3.0f}%'.format(x) for x in vals_util])
            ax_util.set_ylabel('GPU Utilization')
            ax_util.set_xlabel('Time')

            ax_mem.set_yticklabels(['{:3.0f}%'.format(x) for x in vals_mem])
            ax_mem.set_ylabel('GPU Memory Utilization')
            ax_mem.set_xlabel('Time')
            ax_mem.yaxis.set_label_coords(1.07, 0.5)

            plt.draw()
            plt.show()
            plt.pause(0.01)
    except KeyboardInterrupt:
        break
