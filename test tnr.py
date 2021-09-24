import torch
import math
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import numpy as np
import h5py

def beta_function():
    file = h5py.File(".\\data\\TNR beta_function 1.hdf5", "r")
    beta_list = file[("beta_list")]
    lnZ_list = file[("lnZ_list")]
    E_list = file[("E_list")]
    lnZ_exact = file[("lnZ_exact")]
    E_exact = file[("E_exact")]

    E_manual = []
    C_manual = []
    for i in range(1, len(beta_list)-1):
        d_l = beta_list[i] - beta_list[i-1]
        d_r = beta_list[i+1] - beta_list[i]
        E_left = -(lnZ_list[i] - lnZ_list[i - 1]) / d_l
        E_right = -(lnZ_list[i + 1] - lnZ_list[i]) / d_r
        E_manual.append((d_l * E_left + d_r * E_right) / (d_l+d_r))

    for i in range(2, len(E_manual)-1):
        d_l = beta_list[i] - beta_list[i-1]
        d_r = beta_list[i+1] - beta_list[i]

        C_left = -(E_manual[i] - E_manual[i - 1]) / d_l
        C_right = -(E_manual[i + 1] - E_manual[i]) / d_r
        C_manual.append((d_l * C_left + d_r * C_right) / (d_l+d_r))

    fig = plt.figure('TNR')
    fig.subplots_adjust(wspace=0.8,hspace=0.5)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(beta_list, lnZ_exact, label='exact', color='royalblue', alpha=0.8)
    ax1.scatter(beta_list, lnZ_list, marker='+', s=80, label='computed', color='coral', alpha=0.8)
    ax1.set_xlabel(r'$\beta$', fontsize=13)
    ax1.set_ylabel(r'$\ln{Z}$', fontsize=13)
    ax1.tick_params(labelsize=13)
    plt.annotate('(a)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=13, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    ax1.legend(loc='center right', fontsize=12)

    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(beta_list, E_exact, label='exact', color='royalblue', alpha=0.8)
    # ax2.scatter(beta_list, E_list, marker='+', s=80, label='auto differentiation', color='coral', alpha=0.7)
    ax2.scatter(beta_list[1:-1:], E_manual, marker='x', s=40, label='finite-difference', color='coral', alpha=0.7)
    ax2.set_xlabel(r'$\beta$', fontsize=13)
    ax2.set_ylabel(r'internal energy', fontsize=13)
    ax2.tick_params(labelsize=13)
    plt.axvline(x=beta_list[(len(beta_list)-1)/2], c='gray', ls='--',lw=1)
    plt.annotate('(b)', xy=(-0.1, 1.1), xycoords='axes fraction', fontsize=13, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    plt.annotate(r'$\beta_c$', xy=(0.55, 1.0), xycoords='axes fraction', fontsize=13, xytext=(0, 0), textcoords='offset points',ha='right', va='top')
    ax2.legend(loc='upper right', fontsize=12)

    fig2 = plt.figure('TNR2')
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.scatter(beta_list[2:-3:], C_manual, label='diff')


def sc_calc():
    file1 = h5py.File(".\\data\\TNR sc_calc 1.hdf5", "r")
    file2 = h5py.File(".\\data\\TNR sc_calc 2.hdf5", "r")
    file3 = h5py.File(".\\data\\TNR sc_calc 3.hdf5", "r")
    setting1 = file1[("setting")]
    sc1 = file1[("sc")]
    chi1 = setting1[0]
    setting2 = file2[("setting")]
    sc2 = file2[("sc")]
    chi2 = setting2[0]
    setting3 = file3[("setting")]
    sc3 = file3[("sc")]
    chi3 = setting3[0]

    fig = plt.figure('TNR')
    ax_sc = fig.add_subplot(1, 1, 1)
    sc_num = len(sc1)
    ax_sc.scatter(range(sc_num), sc1, marker='o', color='royalblue', s=8, alpha=0.8, label=r"TNR SC ($\chi=$" + str(chi1) + ")")
    sc_num = len(sc2)
    ax_sc.scatter(range(sc_num), sc2, marker='o', color='forestgreen', s=8, alpha=0.8, label=r"TNR SC ($\chi=$" + str(chi2) + ")")
    sc_num = len(sc3)
    ax_sc.scatter(range(sc_num), sc3, marker='o', color='r', s=8, alpha=0.7, label=r"TNR SC ($\chi=$" + str(chi3) + ")")
    for k in range(6):
        ax_sc.plot(range(50), k*np.ones(50), 'k:')
        ax_sc.plot(range(50), k*np.ones(50) + 1/8, 'k:')
    ax_sc.set_ylabel('scaling dimensions', fontsize=13)
    ax_sc.set_xticks([])
    ax_sc.tick_params(labelsize=13)
        
    ax_sc.legend(loc='center right', fontsize=12)

def compare_method():
    file1 = h5py.File(".\\data\\TNR compare_method 1.hdf5", "r")
    method1 = file1[("method")]
    delta1 = file1[("delta_list")]
    file2 = h5py.File(".\\data\\TNR compare_method 2.hdf5", "r")
    method2 = file2[("method")]
    delta2 = file2[("delta_list")]
    setting1 = file1[("setting")]
    totlv = setting1[1]
    epoch = setting1[2]

    fig = plt.figure('loss')
    ax1 = fig.add_subplot(1, 1, 1)

    label = method1[0]
    for i in range(2,totlv,2):
        ax1.loglog(range(0,epoch,2), delta1[i][::2], alpha=0.8, label=label, linestyle='--', color='forestgreen')
        label=None
    label = method2[0]
    for i in range(2,totlv,2):
        ax1.loglog(range(0,epoch,2), delta2[i][::2], alpha=0.6, label=label, linestyle='-', color='purple')
        label=None

    ax1.set_xlabel('iterations', fontsize=13)
    ax1.set_ylabel('approximation errors', fontsize=13)
    ax1.legend(loc='upper right', fontsize=12)

    
if __name__ == "__main__":
    # beta_function()
    sc_calc()
    # compare_method()
    
    plt.show()
