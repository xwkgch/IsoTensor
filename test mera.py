from torch._C import device, dtype
import torch
import math
import matplotlib.pyplot as plt
from model.hamiltonian import Hamiltonian
import numpy as np
import h5py

def g_function_test():
    file = h5py.File(".\\data\\MERA g_function.hdf5", "r")
    g = file[("g")].value
    rho = file[("rho")]
    net = torch.load('.\\data\\MERA g_function.pt')

    device = torch.device('cpu')
    net.to(device)
    dtype = net.ham.dtype
    H = Hamiltonian('Ising', device=device, g=g)
    H.Ising(g=g, requires_grad=True)
    rho = torch.tensor(rho, dtype=dtype, device=device, requires_grad=False)
    rho_0 = net(rho)
    energy = torch.einsum('abcd, abcd', [rho_0, H.ham])

    dE, = torch.autograd.grad(energy, H.g, create_graph=True)
    dE2, = torch.autograd.grad(dE, H.g)
    print(dE,dE2)

def g_function():
    file = h5py.File(".\\data\\MERA g_function 2.hdf5", "r")
    g_list = np.linspace(0.99, 1.01, 21)
    dE_list = file[("dE_list")]

    fig = plt.figure('MERA')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(g_list, dE_list, label='dE', marker='x', color='royalblue', alpha=0.8)

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
    fig1.subplots_adjust(wspace=0.8,hspace=0.5)

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
    ax1.set_xlabel('iterations', fontsize=14)
    ax1.set_ylabel('energy error', fontsize=14)
    ax1.tick_params(labelsize=13)
    plt.annotate('(a)', xy=(-0.1, 2.7), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')

    # ax2.legend(loc='center right', fontsize=12)
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(0, 0, 100), linestyle='-.', color='grey')
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(0.125, 0.125, 100), linestyle='-.', color='grey')
    ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(1, 1, 100), linestyle='-.', color='grey')
    #ax2.plot(np.linspace(0.7, 1.3, 100), np.linspace(1.125, 1.125, 100), linestyle='-.', color='grey')
    ax2.set_xlim([0.7, 1.2])
    ax2.set_ylim([-0.05, 1.2])
    ax2.set_xticks(np.linspace(0.8, 0.8 + 0.1 * 3, len(method_list)))
    ax2.set_yticks([0, 0.125, 1])
    ax2.set_xticklabels(method_list)
    ax2.set_yticklabels(['0', '1/8', '1'])
    ax2.set_ylabel('scaling dimension', fontsize=14)
    ax2.tick_params(labelsize=14)
    plt.annotate('(b)', xy=(-0.1, 1.2), xycoords='axes fraction', fontsize=14, xytext=(0, 0), textcoords='offset points',ha='right', va='top')


if __name__ == "__main__":
    # g_function_test()
    # g_function()
    compare_method()

    plt.show()