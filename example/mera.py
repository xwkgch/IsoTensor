from fileinput import filename
import sys
from time import time
sys.path.append("..")
import torch
import numpy as np
from model.hamiltonian import Hamiltonian
from network.MERAnet import MeraNet
import optim
from tool.monitor import MERAMonitor
from tool.selector import RandomSelect
from lib import functional as func
import h5py

def opt_mix(net, type='Mix'):
    r"""Return a list of optimizations used in random mix method.
    """
    lr0 = 1.0
    opt_list = []
    if type == 'EV':
        opt_list.append(optim.ev.EV(net.parameters(), lr=lr0))
    
    if type == 'Mix':
        opt_list.append(optim.ev.EV(net.parameters(), lr=lr0))
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
        opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    # opt_list.append(optim.adam.Adam(net.parameters(), lr=lr0, betas=(0.9,0.993), amsgrad=True, method='SVD'))
    # opt_list.append(optim.rmsprop.RMSprop(net.parameters(), lr=lr0, momentum=0.9, centered=True, method='SVD'))
    return opt_list

def construct_MERA(model='Ising', chi_list=[4, 6, 7, 8, 9, 10], epoch_list=[200, 500, 1000, 2000, 3000, 4000], g=1.0, totlv=3, info=''):
    r"""An example about construction of a infinite ternary MERA network with random mix method.
    Resetting mechanism is used for Ising model.
    """
    device = torch.device('cuda:0')
    # chi_list = [4, 6, 7, 8, 9, 10, 12]
    # epoch_list = [200, 500, 1000, 2000, 3000, 4000, 4300]
    # chi_list = [8, 9, 10]
    # epoch_list = [2000, 3000, 4000]
    H = Hamiltonian(model, device, g=g)
    net = MeraNet(H.ham, chi=chi_list[0], totlv=totlv).to(device)
    opt_list = opt_mix(net)
    selector = RandomSelect(opt_list)
    selector.add_scheduler('StepLR')
    opt = selector.current
    monitor = MERAMonitor(H, net)
    rho = func.rho_init(net.chi[-1], H.dtype).to(device)

    error_list = []
    time_list = []
    step = res_count = 0
    while step < len(epoch_list):
        chi = chi_list[step]
        if step > 0:
            rho = net.renew(chi, opt_list, rho)

        for epoch in range(epoch_list[step]):    
            rho_0 = net(rho)
            loss = torch.einsum('abcd, abcd', [rho_0, H.ham])
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                rho = func.topdense(rho, list(net.parameters())[-1], list(net.parameters())[-2], func.des_ternary, sciter=4)
            
            monitor.display(loss, opt, container=error_list, timestamp=time_list)
            opt = selector.select()

        step += 1
        if H.model == 'Ising' and step == 2 and monitor.E_error > 1.5e-3:
            net = MeraNet(H.ham, chi=chi_list[0], totlv=3).to(device)
            opt_list = opt_mix(net)
            selector = RandomSelect(opt_list)
            selector.add_scheduler('StepLR')
            opt = selector.current
            monitor = MERAMonitor(H, net)
            rho = func.rho_init(net.chi[-1], H.dtype).to(device)
            error_list = []
            time_list = []
            step = 0
            res_count += 1
            print('\n|-------------------> Restart: %d <--------------------|' % res_count, end='\n')

    monitor.end()

    sc = func.mera_sc(list(net.parameters())[-1])
    print('\n', sc)

    return net, rho

def compare_method(model='Ising'):
    r"""Comparing the performance for different optimization methods.
    """
    device = torch.device('cuda:0')
    chi_list = [4, 6, 7, 8, 9, 10, 12]
    epoch_list = [200, 500, 2000, 3000, 3000, 3000, 4000]
    # chi_list = [4, 5, 6]
    # epoch_list = [200, 500, 1000]
    # chi_list = [4, 6]
    # epoch_list = [30, 70]
    # 
    method_list = ['Evenbly-Vidal', 'SVD', 'Cayley', 'RandomMix']
    totlv = 3
    repeat = 1
    H = Hamiltonian(model, device)

    error_list = [[] for _ in range(len(method_list))]
    time_list = [[] for _ in range(len(method_list))]
    sc_list = []
    
    res_count = 0
    for i in range(len(method_list)):
        net = MeraNet(H.ham, chi=chi_list[0], totlv=totlv).to(device)
        opt_list = opt_mix(net)
        if not i == len(method_list) - 1:
            opt_tmp = [opt_list[i]]
        else:
            opt_tmp = opt_list
        selector = RandomSelect(opt_tmp)
        selector.add_scheduler('StepLR')
        opt = selector.current
        monitor = MERAMonitor(H, net)
        rho = func.rho_init(net.chi[-1], H.dtype).to(device)

        step = 0
        while step < len(epoch_list):
            chi = chi_list[step]
            if step > 0:
                rho = net.renew(chi, opt_list, rho)

            for epoch in range(epoch_list[step]):    
                rho_0 = net(rho)
                loss = torch.einsum('abcd, abcd', [rho_0, H.ham])
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    rho = func.topdense(rho, list(net.parameters())[-1], list(net.parameters())[-2], func.des_ternary, sciter=4)
                
                monitor.display(loss, opt, container=error_list[i], timestamp=time_list[i])
                opt = selector.select()

            step += 1
            if step == 2 and monitor.E_error > 1.5e-3 and H.model == 'Ising':
                net = MeraNet(H.ham, chi=chi_list[0], totlv=totlv).to(device)
                opt_list = opt_mix(net)
                if not i == len(method_list) - 1:
                    opt_tmp = [opt_list[i]]
                else:
                    opt_tmp = opt_list
                selector = RandomSelect(opt_tmp)
                selector.add_scheduler('StepLR')
                opt = selector.current
                monitor = MERAMonitor(H, net)
                rho = func.rho_init(net.chi[-1], H.dtype).to(device)
                error_list[i] = []
                time_list[i] = []
                step = 0
                res_count += 1
                print('\n|-------------------> Restart: %d <--------------------|' % res_count, end='\n')

        sc = func.mera_sc(list(net.parameters())[-1])
        print('\n', sc)
        sc_list.append(sc)
        monitor.end()

    file=h5py.File('.\\data\\MERA compare_method.hdf5',"w")
    file.create_dataset("model", data=np.array([model], dtype=object), dtype=h5py.special_dtype(vlen=str))
    file.create_dataset("method", data=np.array(method_list, dtype=object), dtype=h5py.special_dtype(vlen=str))
    file.create_dataset("setting", data=[totlv, repeat])
    file.create_dataset("epoch", data=epoch_list)
    file.create_dataset("error_list", data=error_list)
    file.create_dataset("time_list", data=time_list)
    file.create_dataset("res_count", data=res_count)
    file.create_dataset("sc_list", data=sc_list)


def single_repeat(model='Ising', chi_list=[4, 6, 7, 8, 9, 10], epoch_list=[200, 500, 1000, 2000, 3000, 4000], g=1.0, totlv=3, info='', mode='cuda', type='Mix', repeat=5):
    r"""Repeat constructing single MERA
    """
    if mode == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    totlv = 3
    # repeat = 5
    H = Hamiltonian(model, device, g=g)

    error_list = [[] for _ in range(repeat)]
    time_list = [[] for _ in range(repeat)]
    sc_list = []
    res_list = []

    threshold = {'Ising':1.5e-3, 'XY':4e-3, 'XXZ': 1.8e-2}
    
    for i in range(repeat):
        net = MeraNet(H.ham, chi=chi_list[0], totlv=totlv).to(device)
        opt_list = opt_mix(net, type)
        selector = RandomSelect(opt_list)
        selector.add_scheduler('StepLR')
        opt = selector.current
        monitor = MERAMonitor(H, net)
        rho = func.rho_init(net.chi[-1], H.dtype).to(device)
        res_count = 0

        step = 0
        while step < len(epoch_list):
            chi = chi_list[step]
            if step > 0:
                rho = net.renew(chi, opt_list, rho)

            for epoch in range(epoch_list[step]):    
                rho_0 = net(rho)
                loss = torch.einsum('abcd, abcd', [rho_0, H.ham])
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    rho = func.topdense(rho, list(net.parameters())[-1], list(net.parameters())[-2], func.des_ternary, sciter=4)
                
                monitor.display(loss, opt, container=error_list[i], timestamp=time_list[i])
                opt = selector.select()

                # if time_list[i][-1] > 800 and epoch > 100:
                #     break

            step += 1
            # if (step == 2 and monitor.E_error > 1.5e-3) or (epoch >= 800 and monitor.E_error > threshold[model]):
            if step == 2 and monitor.E_error > threshold[model]:
                net = MeraNet(H.ham, chi=chi_list[0], totlv=totlv).to(device)
                opt_list = opt_mix(net, type)
                selector = RandomSelect(opt_list)
                selector.add_scheduler('StepLR')
                opt = selector.current
                monitor = MERAMonitor(H, net)
                rho = func.rho_init(net.chi[-1], H.dtype).to(device)
                error_list[i] = []
                time_list[i] = []
                step = 0
                res_count += 1
                print('\n|-------------------> Restart: %d <--------------------|' % res_count, end='\n')

        sc = func.mera_sc(list(net.parameters())[-1])
        print('\n', sc)
        print('--------repeat: %d---------', i)
        sc_list.append(sc)
        monitor.end()
        res_list.append(res_count)

    file_name = '.\\data\\MERA single\\' + model + type + info + str(repeat) + ' ' + mode + '.npz'

    np.savez(file_name, 
        model=np.array([model], dtype=object), 
        setting=np.array([totlv, repeat, g]),
        epoch=np.array(epoch_list),
        chi_list=np.array(chi_list),
        error_list=np.array(error_list),
        time_list=np.array(time_list),
        res_list=np.array(res_list),
        sc_list=np.array(sc_list))

    print('Finish: %s %s %s' % (model, mode, type))
    