from torch._C import device, dtype
from numpy.lib.function_base import select
import torch
import math
import matplotlib.pyplot as plt
from model.hamiltonian import Hamiltonian
import numpy as np
import h5py

def single_repeat():
    file = h5py.File(".\\data\\MERA single\\EVnonlift.hdf5", "r")
    setting = file[("setting")]
    epoch_list = file[("epoch")]
    error_list = file[("error_list")].value

    repeat = int(setting[1])

    fig = plt.figure('MERA')
    ax = fig.add_subplot(1, 1, 1)

    for i in range(repeat):
        ax.loglog(range(0, sum(epoch_list), 5), error_list[i])

    print(file[("sc_list")].value)
    print(file[("time_list")].value)
    print(file[("res_list")].value)

def compare_method():
    file = h5py.File(".\\data\\MERA compare_method 2.hdf5", "r")
    model = file[("model")][0]
    setting = file[("setting")]
    method_list = file[("method")]
    epoch_list = file[("epoch")]
    error_list = file[("error_list")].value
    sc_list = file[("sc_list")].value

    totlv = setting[0]
    repeat = setting[1]
    cut = 3

    #-----------------------------------------------

    file2 = h5py.File(".\\data\\MERA compare_method mix.hdf5", "r")
    error_mix = file2[("error_list")][0]
    sc_mix = file2[("sc_list")][0]
    error_list[-1] = error_mix
    sc_list[-1] = sc_mix
    #-----------------------------------------------

    plot_list = [[0] * repeat for i in range(len(method_list))]
    scatter_list = [[0] * repeat for i in range(len(method_list))]
    fig1 = plt.figure('MERA')
    fig1.subplots_adjust(wspace=0.5,hspace=0.5)

    ax1 = fig1.add_subplot(2, 1, 1)
    ax2 = fig1.add_subplot(2, 1, 2)
    #colors = ['royalblue', 'orange', 'forestgreen']
    #linestyles = ['-', '-', '--']
    colors = ['forestgreen', 'royalblue', 'orange', 'purple']
    linestyles = ['--', '-', '-', '-']
    cut = 3
    
    for i in range(len(method_list)):
        label = method_list[i]

        # for ii in range(repeat):
            # plot_list[i][ii], = ax1.loglog(range(0, sum(epoch_list), 5), error_list[i][ii], alpha=0.8, label=label, linestyle=linestyles[i], color=colors[i])
            # scatter_list[i] = ax2.scatter(np.linspace(0.8, 0.8, cut) + i * 0.1, sc_list[i][ii][:cut], alpha=0.8, s=100, marker='x',label=label, color=colors[i])

            # label = None
        plot_list[i], = ax1.loglog(range(0, sum(epoch_list), 5), error_list[i], alpha=0.8, label=label, linestyle=linestyles[i], color=colors[i])
        scatter_list[i] = ax2.scatter(np.linspace(0.8, 0.8, cut) + i * 0.1, sc_list[i][:cut], alpha=0.8, s=100, marker='x',label=label, color=colors[i])
    
    ax1.legend(loc='upper right', fontsize=12)
    #ax1.set_yscale('log')
    ax1.set_xlabel('Iterations', fontsize=14)
    ax1.set_ylabel('Energy error', fontsize=14)
    ax1.tick_params(labelsize=13)
    plt.annotate('(a)', xy=(-0.1, 2.7), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    # plt.annotate('(a)', xy=(-1.7, 1.1), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    # ax2.legend(loc='center right', fontsize=12)
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(0, 0, 100), linestyle='-.', color='grey')
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(0.125, 0.125, 100), linestyle='-.', color='grey')
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(1, 1, 100), linestyle='-.', color='grey')
    #ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(1.125, 1.125, 100), linestyle='-.', color='grey')
    ax2.set_xlim([0.7, 1.2])
    ax2.set_ylim([-0.05, 1.2])
    ax2.set_xticks(np.linspace(0.8, 0.8 + 0.1 * 3, len(method_list)))
    ax2.set_yticks([0, 0.125, 1])
    ax2.set_xticklabels(method_list,rotation=-15)
    ax2.set_yticklabels(['0', '1/8', '1'])
    ax2.set_ylabel('Scaling dimension', fontsize=14)
    ax2.tick_params(labelsize=14)
    plt.annotate('(b)', xy=(-0.1, 1.2), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    # plt.annotate('(b)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    print(sc_list)

def single_compare():
    file = []
    # file.append(h5py.File(".\\data\\MERA single\\SVDbasic.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVD.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVDnesterov.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVDAdam.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVDRMSprop.hdf5", "r"))

    # file.append(h5py.File(".\\data\\MERA single\\EVnonlift.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVDnonlift.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\Mixnonlift2.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\EV.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\SVD.hdf5", "r"))
    # file.append(h5py.File(".\\data\\MERA single\\Mix2.hdf5", "r"))
    file.append(h5py.File(".\\data\\MERA single\\Mix2 large.hdf5", "r"))
    
    setting = file[0][("setting")]
    repeat = int(setting[1])

    fig = plt.figure('MERA')
    ax = fig.add_subplot(1, 1, 1)
    #, 'royalblue'
    # colors = ['forestgreen', 'orange', 'purple', 'forestgreen', 'orange', 'purple']
    # linestyles = ['--', '--', '--', '-', '-', '-']
    # label_list = ['Evenbly-Vidal without lifting', 'SGD with momentum without lifting', 'RandomMix without lifting', 'Evenbly-Vidal', 'SGD with momentum', 'RandomMix']
    # select_list = [0,1,0,0,1,0]
    linestyles = ['--', '-', '-', '-', '-']
    label_list = ['Basic', 'SGD with momentum', 'SGD with nesterov momentum', 'Adam', 'RMSprop']
    colors = ['forestgreen', 'royalblue', 'orange', 'crimson', 'purple']
    select_list = [0,1,0,0,2]

    for i in range(len(file)):
        error_list = file[i][("error_list")]
        epoch_list = file[i][("epoch")]
        for j in range(repeat):
            ax.loglog(range(0, sum(epoch_list), 5), error_list[j], label=label_list[i], color=colors[i], linestyle=linestyles[i])

        # ax.loglog(range(0, sum(epoch_list), 5), error_list[select_list[i]], label=label_list[i], color=colors[i], linestyle=linestyles[i])
        # print(file[i][("time_list")][select_list[i]])

    ax.legend(loc='upper right', fontsize=12)
    #ax.set_yscale('log')
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Energy error', fontsize=14)
    ax.tick_params(labelsize=13)

def single_compare2():
    file = []
    file.append(h5py.File(".\\data\\MERA single\\EV 7.hdf5", "r"))
    file.append(h5py.File(".\\data\\MERA single\\Mix 7.hdf5", "r"))
    setting = file[0][("setting")]
    repeat = int(setting[1])

    fig = plt.figure('MERA')
    ax = fig.add_subplot(1, 1, 1)

    linestyles = ['--', '-', '-', '-', '-']
    label_list = ['Basic', 'SGD with momentum', 'SGD with nesterov momentum', 'Adam', 'RMSprop']
    colors = ['forestgreen', 'royalblue', 'orange', 'crimson', 'purple']

    for i in range(len(file)):
        error_list = file[i][("error_list")]
        time_list = file[i][("time_list")]
        for j in range(repeat):
            if not j:
                ax.loglog(time_list[j], error_list[j], label=label_list[i], color=colors[i], linestyle=linestyles[i])
            else:
                ax.loglog(time_list[j], error_list[j], color=colors[i], linestyle=linestyles[i])

    ax.legend(loc='upper right', fontsize=12)
    #ax.set_yscale('log')
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Energy error', fontsize=14)
    ax.tick_params(labelsize=13)

if __name__ == "__main__":
    # single_repeat()
    # compare_method()
    # single_compare()
    single_compare2()
    r = np.load('.\\data\\MERA single\\test.npz',allow_pickle=True)
    print(r['tl'][0])

    plt.show()