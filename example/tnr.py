import sys
sys.path.append("..")
import torch
import numpy as np
from model.nettensor import NetTensor
from network.TNRnet import TNRNet
import optim
import layer.TNRlayer as tnrlayer
from tool.monitor import TNRMonitor
from tool.selector import RandomSelect
from lib import functional as func
import h5py
import itertools

def opt_mix(net, type='Mix', lr0 = 1.0):
    r"""Return a list of optimizations used in random mix method.
    """
    opt_list = []
    if type == 'EV':
        opt_list.append(optim.ev.EV(net.parameters(), lr=lr0))
    
    if type == 'Mix':
        opt_list.append(optim.ev.EV(net.parameters(), lr=lr0))
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    # opt_list.append(optim.adam.Adam(net.parameters(), lr=lr0, betas=(0.9,0.993), amsgrad=True, method='SVD'))
    # opt_list.append(optim.rmsprop.RMSprop(net.parameters(), lr=lr0, momentum=0.9, centered=True, method='SVD'))
    if type == 'SGD':
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    return opt_list

def opt_mix2(net, net2, type='Mix', lr0 = 1.0):
    r"""Return a list of optimizations used in random mix method.
    """
    opt_list = []
    if type == 'EV':
        opt_list.append(optim.ev.EV(itertools.chain(net.parameters(), net2.parameters()), lr=lr0))
    
    if type == 'Mix':
        opt_list.append(optim.ev.EV(itertools.chain(net.parameters(), net2.parameters()), lr=lr0))
        opt_list.append(optim.sgd.SGD(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
        opt_list.append(optim.sgd.SGD(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    # opt_list.append(optim.adam.Adam(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, betas=(0.9,0.993), amsgrad=True, method='SVD'))
    # opt_list.append(optim.rmsprop.RMSprop(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, momentum=0.9, centered=True, method='SVD'))
    if type == 'SGD':
        opt_list.append(optim.sgd.SGD(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
        opt_list.append(optim.sgd.SGD(itertools.chain(net.parameters(), net2.parameters()), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    return opt_list

def construct_TNR(beta=None, chi=8, totlv=8, epoch=400):
    r"""An example about construction a TNR coarse-graining and optimization networks with random mix method.
    Autograd values are computed. (problematic)
    """
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda:0')
    totlv = totlv
    chi = chi
    chi_list = [chi] * 4
    if beta:
        model = NetTensor('CIsing',beta=beta)
    else:
        model = NetTensor('CIsing')
    net = TNRNet(model, chi_list=chi_list, totlv=totlv).to(device)
    monitor = TNRMonitor(model, net)
    delta_list = []

    beta = torch.tensor(model.beta0, requires_grad=False)

    # for i in range(totlv):
    #     opt_list = opt_mix(net.layers_tnr[i], 'Mix')
    #     selector = RandomSelect(opt_list)
    #     opt = selector.current

    #     delta_list.append([])

    #     for _ in range(epoch):
    #         loss = net.forward(i)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #         delta_list[i].append(loss.item())
    #         monitor.display(loss, i)
    #         opt = selector.select()
    #         with torch.no_grad():
    #             net.RG_step(i)
    #             print('   %f' % net.layers_div[i+1].norm, end='')

    #     monitor.nextlv()
        
    #     with torch.no_grad():
    #         net.RG_step(i)

    # for i in range(totlv):
    #     with torch.no_grad():
    #         net.RG_step(i)
    #         print('   %f' % net.layers_div[i+1].norm, end='')
    # print('\n------')
    # for i in range(totlv-1):
    #     with torch.no_grad():
    #         net.forward_trans(i)
    #         print('   %f' % net.layers_div[i+1].norm, end='')

    # with torch.no_grad():
    #     lnZ = net.sum_lnZ().item()

    # error = abs(lnZ - model.lnZ_exact)
    # print(error)
    # sc = func.tnr_sc(net.A_list[-1], 4)
    # print(sc)
    

    for i in range(0, totlv, 2):
        if i == totlv-1:
            opt_list = opt_mix(net.layers_tnr[i], 'EV', lr0=0.00)
        else:
            opt_list = opt_mix2(net.layers_tnr[i], net.layers_tnr[i+1], 'SGD', lr0=0.0001)
        selector = RandomSelect(opt_list, stride=2)
        selector.add_scheduler('StepLR')
        opt = selector.current

        delta_list.append([])

        for _ in range(epoch):
            if i == totlv-1:
                loss = net.forward(i)
            else:
                loss = net.forward_trans(i)
            # loss = net.forward(i)
            opt.zero_grad()
            loss.backward()
            opt.step()

            delta_list[i].append(loss.item())
            monitor.display(loss, i)
            opt = selector.select()

            # with torch.no_grad():
                # net.RG_step(i)
            # for j in range(totlv):
            #     print('   %f' % net.layers_div[j+1].norm, end='')
        monitor.nextlv()
        
        with torch.no_grad():
            net.RG_step_trans(i)
            net.update_exact(i+1)
            # print(net.forward(i))
            # net.RG_step(i)
            # net.update_exact(i)
            delta_list.append([])
            # print(net.forward(i+1))
            # net.RG_step(i+1)
            # net.update_exact(i+1)
            # print('=>', end='')
            # for j in range(totlv):
            #     print('   %f' % net.layers_div[j+1].norm, end='')
            # print('')

    lnZ = net.sum_lnZ_trans().item()

    monitor.end()
    error = abs(lnZ - model.lnZ_exact)
    print(error)
    sc = func.tnr_sc(net.A_list[-1], 4)
    print(sc[:6])

    import matplotlib.pyplot as plt
    fig = plt.figure('TNR')
    ax = fig.add_subplot(1, 1, 1)
    for i in range(0, totlv, 2):
        ax.loglog(range(0, epoch, 1), delta_list[i], label=str(i))
    plt.show()

    return model, lnZ, net, delta_list

def b_list(d1, d2, d3, n1, n2, n3):
    beta_c = np.log(1 + np.sqrt(2)) / 2
    b_c = np.linspace(beta_c-n1*d1, beta_c+n1*d1, 2*n1+1)
    b_l1 = np.linspace(beta_c-n1*d1-n2*d2, beta_c-n1*d1-d2, n2)
    b_l2 = np.linspace(beta_c-n1*d1-n2*d2-n3*d3, beta_c-n1*d1-n2*d2-d3, n3)
    b_r1 = np.linspace(beta_c+n1*d1+d2, beta_c+n1*d1+n2*d2, n2)
    b_r2 = np.linspace(beta_c+n1*d1+n2*d2+d3, beta_c+n1*d1+n2*d2+n3*d3, n3)

    return np.concatenate((b_l2,b_l1,b_c,b_r1,b_r2))

def beta_function():
    r"""Compute the lnZ by TNR networks and internal energy by finite-differentiation as a function of beta.
    """
    beta_list = b_list(0.001,0.002,0.003,10,8,6)
    print(beta_list)
    file=h5py.File('.\\data\\TNR beta_function.hdf5',"w")
    file.create_dataset("beta_list", data=beta_list)

    lnZ_list = []
    E_list = []
    lnZ_exact = []
    E_exact = []
    
    for b in beta_list:
        print('beta = %f' % b)
        ten, lnZ_tmp, E_tmp, _, _ = construct_TNR(b, chi=14)
        lnZ_exact.append(ten.lnZ_exact)
        E_exact.append(ten.E_exact)
        lnZ_list.append(lnZ_tmp)
        E_list.append(E_tmp)

    file.create_dataset("lnZ_list", data=lnZ_list)
    file.create_dataset("E_list", data=E_list)
    file.create_dataset("lnZ_exact", data=lnZ_exact)
    file.create_dataset("E_exact", data=E_exact)

def sc_calc():
    r"""Compute the scaling dimensions for TNR."""
    file=h5py.File('.\\data\\TNR sc_calc.hdf5',"w")
    chi = 14
    totlv = 10
    epoch = 800
    file.create_dataset("setting", data=[chi, totlv, epoch])

    ten, lnZ_tmp, E_tmp, net, _ = construct_TNR(None, chi=chi, totlv=totlv, epoch=epoch)
    torch.save(net, '.\\data\\tnr_sc chi' + str(chi) + '.pt')
    beta = torch.tensor(ten.beta0, requires_grad=True)
    A_top = net(beta)
    sc = func.tnr_sc(A_top, 4)
    print(sc)
    file.create_dataset("sc", data=sc)

def compare_method():
    r"""Comparing the performance for different optimization methods.
    """
    file=h5py.File('.\\data\\TNR compare_method.hdf5',"w")
    chi = 14
    totlv = 8
    epoch = 1000
    method = 'Evenbly-Vidal'
    file.create_dataset("method", data=np.array([method], dtype=object), dtype=h5py.special_dtype(vlen=str))
    file.create_dataset("setting", data=[chi, totlv, epoch])

    ten, lnZ_tmp, E_tmp, net, delta_list = construct_TNR(None, chi=chi, totlv=totlv, epoch=epoch)
    file.create_dataset("delta_list", data=delta_list)