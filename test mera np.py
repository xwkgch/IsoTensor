from matplotlib import markers
from torch._C import device, dtype
from numpy.lib.function_base import select
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def plot(ax, x, y, type='plot', color=None, linestyle='-', label=None, alpha=1.0, marker=None, s=None):
    if type == 'plot':
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha)
    elif type == 'log':
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha)
        ax.set_yscale('log')
    elif type == 'loglog':
        ax.loglog(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha)
    elif type == 'scatter':
        ax.scatter(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, marker=marker, s=s)
    elif type == 'scatter log':
        ax.scatter(x, y, label=label, color=color, linestyle=linestyle, alpha=alpha, marker=marker, s=s)
        ax.set_yscale('log')

def plot_set(ax, xlabel=None, ylabel=None, loc='upper right'):
    ax.legend(loc=loc, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=13)

def time_cost(epoch_list, time_list, log=False):
    yy = []
    for epoch in np.cumsum(epoch_list):
        ind = int(epoch/5) - 1
        if len(yy):
            tmp = (time_list[ind] - yy[-1]) / epoch
        else:
            tmp = (time_list[ind]) / epoch
        if log:
            tmp = math.log(tmp)
        yy.append(tmp)
    return yy

def data_combine(K=15, mode='EV cpu'):
    data = []
    if mode == 'EV cpu':
        data.append(np.load('.\\data\\MERA single\\EV 9 cpu 0.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\EV 9 cpu 1.npz',allow_pickle=True))
    elif mode == 'EV cuda':
        data.append(np.load('.\\data\\MERA single\\EV 9 cuda 0.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\EV 7 cuda 1.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\EV 5 cuda 2.npz',allow_pickle=True))
    elif mode == 'Mix cuda':
        data.append(np.load('.\\data\\MERA single\\Mix 9 cuda 0.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\Mix 7 cuda 0.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\Mix 5 cuda 0.npz',allow_pickle=True))
    elif mode == 'Conventional':
        data.append(np.load('.\\data\\MERA single\\Conventional 0.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\Conventional 1.npz',allow_pickle=True))
    elif mode == 'XY EV cpu':
        data.append(np.load('.\\data\\MERA single\\XYEV 9 cpu.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XYEV 5 cpu.npz',allow_pickle=True))
    elif mode == 'XY EV cuda':
        data.append(np.load('.\\data\\MERA single\\XYEV 9 cuda.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XYEV 5 cuda.npz',allow_pickle=True))
        # data.append(np.load('.\\data\\MERA single\\XYEV 4 cuda.npz',allow_pickle=True))
    elif mode == 'XY Mix cuda':
        data.append(np.load('.\\data\\MERA single\\XYMix 9 cuda.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XYMix 5 cuda.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XYMix 8 cuda.npz',allow_pickle=True))
    elif mode == 'XXZ EV cpu':
        data.append(np.load('.\\data\\MERA single\\XXZEV 9 cpu.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZEV 5 cpu.npz',allow_pickle=True))
    elif mode == 'XXZ EV cuda':
        data.append(np.load('.\\data\\MERA single\\XXZEV 9 cuda.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZEV 5 cuda.npz',allow_pickle=True))
    elif mode == 'XXZ Mix cuda':
        data.append(np.load('.\\data\\MERA single\\XXZMix 9 cuda.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZMix 5 cuda.npz',allow_pickle=True))

    errors = []
    for i in range(len(data)):
        setting = data[i]["setting"]
        repeat = int(setting[1])
        for j in range(repeat):
            errors.append((data[i]['error_list'][j][-1], i, j))
    errors_sort = sorted(errors)

    
    setting[1] = K
    epoch_list = data[0]["epoch"]
    chi_list = data[0]["chi_list"]
    error_list = []
    time_list = []
    res_list = []
    sc_list = []
    for k in range(K):
        i = errors_sort[k][1]
        j = errors_sort[k][2]
        error_list.append(data[i]['error_list'][j])
        time_list.append(data[i]['time_list'][j])
        sc_list.append(sorted(data[i]['sc_list'][j]))
        if 'res_list' in data[i].keys():
            model = data[0]["model"][0]
            res_list.append(data[i]['res_list'][j])
            
    file_name = '.\\data\\MERA single\\' + mode + ' final.npz'

    if 'res_list' in data[0].keys():
        np.savez(file_name, 
        model=np.array([model], dtype=object), 
        setting=np.array(setting),
        epoch=np.array(epoch_list),
        chi_list=np.array(chi_list),
        error_list=np.array(error_list),
        time_list=np.array(time_list),
        res_list=np.array(res_list),
        sc_list=np.array(sc_list))
    else:
        np.savez('.\\data\\MERA single\\Conventional final.npz', 
        setting=np.array(setting),
        epoch=np.array(epoch_list),
        chi_list=np.array(chi_list),
        error_list=np.array(error_list),
        time_list=np.array(time_list),
        sc_list=np.array(sc_list))

def single_compare(model = 'Ising'):
    data = []
    if model == 'Ising':
        data.append(np.load('.\\data\\MERA single\\Conventional final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\EV cpu final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\EV cuda final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\Mix cuda final.npz',allow_pickle=True))
    elif model == 'XY':
        data.append(np.load('.\\data\\MERA single\\XY Conventional 3.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XY EV cpu final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XY EV cuda final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XY Mix cuda final.npz',allow_pickle=True))
    elif model == 'XXZ':
        data.append(np.load('.\\data\\MERA single\\XXZ Conventional 3.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZ EV cpu final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZ EV cuda final.npz',allow_pickle=True))
        data.append(np.load('.\\data\\MERA single\\XXZ Mix cuda final.npz',allow_pickle=True))

    fig1 = plt.figure('MERA', figsize=(7,5), tight_layout=True)
    fig2 = plt.figure('MERA loglog', figsize=(7,5), tight_layout=True)
    fig3 = plt.figure('complexity', figsize=(7,5), tight_layout=True)
    fig4 = plt.figure('Scaling dimensions', figsize=(7,5), tight_layout=True)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax4 = fig4.add_subplot(1, 1, 1)

    linestyles = ['-', '-', '-', '-', '-']
    label_list = ['Conventional', 'EV CPU', 'EV GPU', 'Mix GPU']
    colors = ['forestgreen', 'royalblue', 'coral', 'mediumpurple', 'crimson']
    ax3_type = 'scatter'
    ax3_log = True
    if ax3_log:
        ax3_type = ax3_type + ' log'

    for i in range(len(data)):
        setting = data[i]["setting"]
        repeat = int(setting[1])
        error_list = data[i]["error_list"]
        time_list = data[i]["time_list"]
        epoch_list = data[i]["epoch"]
        chi_list = data[i]["chi_list"]
        error_tmp = []
        time_tmp = []
        sc_tmp = []
        sc = False
        if 'sc_list' in data[i].keys():
            sc = True
            sc_list = data[i]["sc_list"]
            cut = 5
            # print(sc_list[:5])
        for j in range(repeat):
            yy = time_cost(epoch_list, time_list[j])
            if not j:
                plot(ax1, time_list[j], error_list[j], type='log', label=label_list[i], color=colors[i], linestyle=linestyles[i], alpha=0.2)
                plot(ax2, time_list[j], error_list[j], type='loglog', label=label_list[i], color=colors[i], linestyle=linestyles[i], alpha=0.2)
                plot(ax3, chi_list, yy, type=ax3_type, label=label_list[i], color=colors[i], linestyle=linestyles[i], alpha=0.15, marker='o', s=10)
                if sc:
                    plot(ax4, np.linspace(0.8, 0.8, cut) + i * 0.1, sc_list[j][:cut], type='scatter', label=label_list[i], color=colors[i], linestyle=linestyles[i], alpha=0.15, marker='o', s=10)
            else:
                plot(ax1, time_list[j], error_list[j], type='log', color=colors[i], linestyle=linestyles[i], alpha=0.2)
                plot(ax2, time_list[j], error_list[j], type='loglog', color=colors[i], linestyle=linestyles[i], alpha=0.2)
                plot(ax3, chi_list, yy, type=ax3_type, color=colors[i], linestyle=linestyles[i], alpha=0.15, marker='o', s=10)
                if sc:
                    plot(ax4, np.linspace(0.8, 0.8, cut) + i * 0.1, sc_list[j][:cut], type='scatter', color=colors[i], linestyle=linestyles[i], alpha=0.15, marker='o', s=10)
            error_tmp.append(error_list[j])
            time_tmp.append(time_list[j])
            if sc:
                sc_tmp.append(sc_list[j])

        error_avg = np.average(error_tmp, axis=0)
        time_avg = np.average(time_tmp, axis=0)
        yy = time_cost(epoch_list, time_avg)
        plot(ax1, time_avg, error_avg, type='log', label=label_list[i] + ' avg', color=colors[i], linestyle=linestyles[i], alpha=1.0)
        plot(ax2, time_avg, error_avg, type='loglog', label=label_list[i] + ' avg', color=colors[i], linestyle=linestyles[i], alpha=1.0)
        plot(ax3, chi_list, yy, type=ax3_type, label=label_list[i] + ' avg', color=colors[i], linestyle=linestyles[i], alpha=0.8, marker='x', s=50)
        if sc:
            sc_avg = np.average(sc_tmp, axis=0)
            plot(ax4, np.linspace(0.8, 0.8, cut) + i * 0.1, sc_avg[:cut], type='scatter', label=label_list[i] + ' avg', color=colors[i], linestyle=linestyles[i], alpha=0.8, marker='x', s=50)
    
    plot_set(ax1, 'Time (seconds)', 'Relative energy error')
    plot_set(ax2, 'Time (seconds)', 'Relative energy error')
    plot_set(ax3, 'Bond dimension', 'Time cost per iteration', loc='upper left')
    plot_set(ax4, 'methods', 'Scaling dimensions', loc='center right')
    ax4.plot(np.linspace(0.7, 1.2, 100), np.linspace(0, 0, 100), linestyle='-.', color='grey')
    ax4.plot(np.linspace(0.7, 1.2, 100), np.linspace(0.125, 0.125, 100), linestyle='-.', color='grey')
    ax4.plot(np.linspace(0.7, 1.2, 100), np.linspace(1, 1, 100), linestyle='-.', color='grey')
    ax4.plot(np.linspace(0.7, 1.2, 100), np.linspace(1.125, 1.125, 100), linestyle='-.', color='grey')
    # ax4.plot(np.linspace(0.7, 1.2, 100), np.linspace(2, 2, 100), linestyle='-.', color='grey')
    ax4.set_xlim([0.7, 1.2])
    ax4.set_ylim([-0.05, 1.2])
    ax4.set_xticks(np.linspace(0.8, 0.8 + 0.1 * 3, len(label_list)))
    ax4.set_yticks([0, 0.125, 1, 1.125])
    ax4.set_xticklabels(label_list,rotation=0)
    ax4.set_yticklabels(['0', '0.125', '1', '1.125'],rotation=0)
        

if __name__ == "__main__":
    # data_combine(K=12, mode='EV cpu')
    # data_combine(K=12, mode='EV cuda')
    # data_combine(K=12, mode='Mix cuda')
    # data_combine(K=6, mode='Conventional')
    # data_combine(K=7, mode='XY EV cpu')
    # data_combine(K=7, mode='XY EV cuda')
    # data_combine(K=7, mode='XY Mix cuda')
    # data_combine(K=6, mode='XXZ EV cpu')
    # data_combine(K=6, mode='XXZ EV cuda')
    # data_combine(K=6, mode='XXZ Mix cuda')
    # single_compare('XY')
    single_compare('XXZ')

    plt.show()