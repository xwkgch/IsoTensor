import sys
sys.path.append("..")
import torch
from lib import functional as func
from layer import MERAlayer

class MeraNet(torch.nn.Module):
    r"""A standard infinite ternary MERA network.
    Args:
        ham (tensor): two-site Hamiltonian tensor
        chi (int, optional): the max bond dimension (default: 6)
        totlv (int, optional): the number of (transitional) layers (default: 3)
    """
    def __init__(self, ham, chi=6, totlv=3):
        super().__init__()
        self.ham = ham
        self.totlv = totlv
        self.norm_list = []
        
        self.chi = [0] * (totlv + 1)
        self.chi[0] = ham.size(0)
        for i in range(totlv):
            self.chi[i + 1] = min(chi, self.chi[i]** 3)

        layers = [MERAlayer.SimpleTernary(self.chi[i], self.chi[i+1], (3,1), ham.dtype) for i in range(totlv)]
        self.net = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in reversed(self.net):
            x = layer(x)
        return x

    def renew(self, chi_new, opt_list, rho):
        r"""Expand the bond dimension.
        Args:
            chi_new (int): new bond dimension
            opt (optimizer): optimizer associated with this network
            rho (Tensor): the density tensor to be expanded
        Return (Tensor): the expanded density tensor
        """
        print('')
        assert chi_new >= self.chi[-1]

        param_groups_old = list(self.parameters())

        for i in range(self.totlv):
            self.chi[i + 1] = min(chi_new, self.chi[i]** 3)
            self.net[i].padding(self.chi[i], self.chi[i+1])

        for opt in opt_list:
            # opt.__init__(self.parameters(), **{k:v for k,v in opt.param_groups[0].items() if k in opt.defaults})
            param_groups = list(self.parameters())
            for i, param in enumerate(param_groups):
                opt.param_groups[0]['params'][i] = param
                p_old = param_groups_old[i]
                for item in opt.pad_item:
                    opt.state[param][item] = func.pad(opt.state[p_old][item], param.size())
        return func.rho_init(chi_new, rho=rho)

    