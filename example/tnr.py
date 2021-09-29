import sys
sys.path.append("..")
import torch
import numpy as np
from model.nettensor import NetTensor
from network.TNRnet import TNRNet
import optim
import layer.TNRlayer as ntrlayer
from tool.monitor import TNRMonitor
from tool.selector import RandomSelect
from lib import functional as func
import h5py

def opt_mix(net):
    r"""Return a list of optimizations used in random mix method.
    """
    opt_list = []
    opt_list.append(optim.ev.EV(net.parameters(), lr=0.6))
    opt_list.append(optim.sgd.SGD(net.parameters(), lr=0.6, momentum=0.9, nesterov=False, method='SVD'))
    opt_list.append(optim.sgd.SGD(net.parameters(), lr=0.6, momentum=0.9, nesterov=False, method='Cayley'))
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
        ten = NetTensor('CIsing',beta=beta)
    else:
        ten = NetTensor('CIsing')
    net = TNRNet(ten.chi_HV, chi_list=chi_list, dtype=ten.dtype, totlv=totlv).to(device)
    opt_net = torch.nn.ModuleList([ntrlayer.LayerTNROpt(net.layers_tnr[i].chi_HV, chi_list, ten.A.dtype) for i in range(totlv)])
    monitor = TNRMonitor(ten, net)
    delta_list = []

    beta = torch.tensor(ten.beta0, requires_grad=False)
    for i in range(totlv):
        A_top = net(beta)

        opt_list = opt_mix(opt_net[i])
        selector = RandomSelect(opt_list)
        opt = selector.current

        delta_list.append([])

        for _ in range(epoch):
            loss = opt_net[i](net.layers_tnr[i].buf_A, net.layers_tnr[i].buf_A)
            opt.zero_grad()
            loss.backward()
            opt.step()
            net.layers_tnr[i].set_param(opt_net[i].get_param())

            delta_list[i].append(loss.item())
            monitor.display(loss, i)
            opt = selector.select()
        monitor.nextlv()

    beta = torch.tensor(ten.beta0, requires_grad=True)
    A_top = net(beta)
    lnZ = net.sum_lnZ(A_top)
    dlnZ, = torch.autograd.grad(lnZ, beta, create_graph=True)
    dlnZ2, = torch.autograd.grad(dlnZ, beta)
    print(lnZ.item()-ten.lnZ_exact, dlnZ.item()+ten.E_exact)

    monitor.end()

    return ten, lnZ.item(), -dlnZ.item(), net, delta_list

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