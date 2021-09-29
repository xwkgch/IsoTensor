import sys
sys.path.append("..")
import torch
from model.hamiltonian import Hamiltonian
from network.MERAnet import MeraNet
import optim
from tool.monitor import MERAMonitor
from lib import functional as func
import h5py
"""This file is problematic."""
def ad_Iing(g, dtype, device):
    sX = torch.tensor([[0, 1.0], [1.0, 0]], dtype=dtype, device=device)
    sZ = torch.tensor([[1.0, 0], [0, -1.0]], dtype=dtype, device=device)
    H = torch.einsum('xz, wy -> zyxw', sX, sX) + 0.5 * g * (torch.einsum('xz, wy -> zyxw', sZ, torch.eye(2, dtype=dtype, device=device)) + torch.einsum('xz, wy -> zyxw', torch.eye(2, dtype=dtype, device=device), sZ))
    H = 0.5 * (H + H.permute(1, 0, 3, 2))
    return H

def construct_MERA_simple():
    device = torch.device('cuda:0')
    chi = 6
    epoch = 20
    
    H = Hamiltonian('Ising', device, g=1.01)
    gt = torch.tensor(1.01, dtype=H.dtype, device=H.device, requires_grad=True)
    ham = ad_Iing(gt, dtype=H.dtype, device=H.device)

    net = MeraNet(H.ham, chi=chi, totlv=3).to(device)
    opt = optim.ev.EV(net.parameters(), lr=1.0)
    monitor = MERAMonitor(H, net)
    rho = func.rho_init(net.chi[-1], H.dtype).to(device)

    loss_list = []

    for _ in range(epoch):    
        rho_0 = net(rho)
        loss = torch.einsum('abcd, abcd', [rho_0, ham])
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()

        with torch.no_grad():
            rho = func.topdense(rho, list(net.parameters())[-1], list(net.parameters())[-2], func.des_ternary, sciter=4)
        
        monitor.display(loss, opt)
        loss_list.append(loss)

    monitor.end()

    dE = torch.autograd.grad(loss_list[-1], gt, create_graph=True)
    print(dE)
    ddE = torch.autograd.grad(dE, gt)
    print(ddE)