import sys
sys.path.append("..")
import torch
from lib import functional as func
from layer import TNRlayer

class TNRNet(torch.nn.Module):
    r"""A standard TNR coarse-graining network.
    """
    def __init__(self, chi_HV, chi_list=(8,8,8,8), dtype=torch.double, totlv=8):
        super().__init__()
        self.chi_list = chi_list
        self.totlv = totlv
        self.chi_HV = chi_HV
        self.dtype = dtype

        self.layers_tnr = []
        layers = [TNRlayer.LayerGen(), TNRlayer.LayerDiv()]
        for i in range(totlv):
            self.layers_tnr.append(TNRlayer.LayerTNR(chi_HV, chi_list, dtype))
            layers.append(self.layers_tnr[-1])
            layers.append(TNRlayer.LayerDiv())
            _,_,_,_, chiAH, chiAV = func.get_chi(chi_HV, chi_list)
            chi_HV = (chiAH, chiAV)

        self.net = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

    def sum_lnZ(self, A_top):
        r"""Compute the lnZ at the top layer.
        """
        lnZ = torch.log(torch.einsum('abab', A_top))
        i = 0
        for lay in reversed(self.net):
            if isinstance(lay, TNRlayer.LayerDiv):
                lnZ += 4 ** i * torch.log(lay.norm) 
                i += 1
        lnZ = lnZ / 4 ** (self.totlv) 

        return lnZ
