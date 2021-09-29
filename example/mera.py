import sys
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

def opt_mix(net):
    r"""Return a list of optimizations used in random mix method.
    """
    lr0 = 1.0
    opt_list = []
    opt_list.append(optim.ev.EV(net.parameters(), lr=lr0))
    opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='SVD'))
    opt_list.append(optim.sgd.SGD(net.parameters(), lr=lr0, momentum=0.9, nesterov=False, method='Cayley'))
    # opt_list.append(optim.adam.Adam(net.parameters(), lr=lr0, betas=(0.9,0.999), amsgrad=True, method='SVD'))
    return opt_list

def construct_MERA(model='Ising', chi_list=[4, 6, 7, 8], epoch_list=[300, 700, 1000, 2000], g=1.0):
    r"""An example about construction of a infinite ternary MERA network with random mix method.
    Resetting mechanism is used for Ising model.
    """
    device = torch.device('cuda:0')
    chi_list = [4, 6, 7, 8, 9, 10]
    epoch_list = [300, 700, 1000, 2000, 3000, 4000]
    H = Hamiltonian(model, device, g=g)
    net = MeraNet(H.ham, chi=chi_list[0], totlv=3).to(device)
    opt_list = opt_mix(net)
    selector = RandomSelect(opt_list)
    selector.add_scheduler('StepLR')
    opt = selector.current
    monitor = MERAMonitor(H, net)
    rho = func.rho_init(net.chi[-1], H.dtype).to(device)

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
            
            monitor.display(loss, opt)
            opt = selector.select()

        step += 1
        if step == 2 and monitor.E_error > 1e-3 and H.model == 'Ising':
            net = MeraNet(H.ham, chi=chi_list[0], totlv=3).to(device)
            opt_list = opt_mix(net)
            selector = RandomSelect(opt_list)
            selector.add_scheduler('StepLR')
            opt = selector.current
            monitor = MERAMonitor(H, net)
            rho = func.rho_init(net.chi[-1], H.dtype).to(device)
            step = 0
            res_count += 1
            print('\n|-------------------> Restart: %d <--------------------|' % res_count, end='\n')

    monitor.end()

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
                
                monitor.display(loss, opt, container=error_list[i])
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
    file.create_dataset("setting", data=[3, 1])
    file.create_dataset("epoch", data=epoch_list)
    file.create_dataset("error_list", data=error_list)
    file.create_dataset("res_count", data=res_count)
    file.create_dataset("sc_list", data=sc_list)


# def g_function():
#     device = torch.device('cuda:0')
#     file=h5py.File('.\\data\\MERA g_function.hdf5',"w")
#     g_list = np.linspace(0.99, 1.01, 21)
#     g = 1.001
#     model='Ising'
#     chi_list=[4, 5, 6, 7, 8]
#     epoch_list=[300, 700, 10, 20, 30]
#     # epoch_list=[200, 500, 20, 40]
#     file.create_dataset("g_list", data=g_list)
#     dE_list = []

#     for g in g_list:
#         net, rho = construct_MERA(model,chi_list,epoch_list,g=g)
#         # torch.save(net, '.\\data\\MERA g_function.pt')
#         # file.create_dataset("rho", data=rho.cpu().detach().numpy())
#         H = Hamiltonian('Ising', device=device, g=g)

#         gt = torch.tensor(g, device=device, dtype=H.dtype, requires_grad=True)
#         sX = torch.tensor([[0, 1.0], [1.0, 0]], dtype=H.dtype, device=device)
#         sZ = torch.tensor([[1.0, 0], [0, -1.0]], dtype=H.dtype, device=device)
#         ham = torch.einsum('xz, wy -> zyxw', sX, sX) + 0.5 * gt * (torch.einsum('xz, wy -> zyxw', sZ, torch.eye(2, dtype=H.dtype, device=device)) + torch.einsum('xz, wy -> zyxw', torch.eye(2, dtype=H.dtype, device=device), sZ))
#         # ham = 0.5 * (ham + ham.permute(1, 0, 3, 2))
#         # ham = ham - H.bias * torch.eye(4, dtype=H.dtype, device=device).reshape(2, 2, 2, 2)

#         rho_0 = net(rho)
#         energy = torch.einsum('abcd, abcd', [rho_0, ham])

#         dE, = torch.autograd.grad(energy, gt, create_graph=True)
#         dE2, = torch.autograd.grad(dE, gt)
#         # H.g.backward()
#         dE_list.append(dE.item())
#         print(g, dE.item())

#     file.create_dataset("dE_list", data=dE_list)
