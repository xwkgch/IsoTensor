import sys
sys.path.append("..")
import torch
from lib import functional as func
from layer import TNRlayer

class TNRNet(torch.nn.Module):
    r"""A standard TNR coarse-graining network.
    """
    def __init__(self, model, chi_list=(8,8,8,8), totlv=8):
        super().__init__()
        self.chi_list = chi_list
        self.totlv = totlv
        self.A_list = [0] * (self.totlv + 1)
        self.A_origin = model.A
        self.A_exact = [0] * (self.totlv + 1)
        
        self.layers_div = [TNRlayer.LayerDiv()]
        self.layers_tnr = []

        A_in = self.layers_div[0](model.A)
        self.A_list[0] = A_in
        self.A_exact[0] = func.contract_A_exact(self.A_list[0],type='16')

        chi_HV = model.chi_HV
        for i in range(totlv):
            self.layers_tnr.append(TNRlayer.LayerTNR(chi_HV, chi_list, model.dtype))
            self.layers_div.append(TNRlayer.LayerDiv())
            _,_,_,_, chiAH, chiAV = func.get_chi(chi_HV, chi_list)
            chi_HV = (chiAH, chiAV)  

    def RG_step(self, lay):
        A_in = self.A_list[lay]
        A_out = self.layers_tnr[lay].RG_step(A_in)
        A_in = self.layers_div[lay + 1](A_out)
        self.A_list[lay + 1] = A_in
        return A_in

    def RG_step_trans(self, lay):
        A_in = self.A_list[lay]
        A_out = self.layers_tnr[lay].RG_step(A_in)
        A_out2 = self.layers_tnr[lay + 1].RG_step(A_out)
        A_in = self.layers_div[lay + 2](A_out2)
        self.A_list[lay + 2] = A_in
        return A_in

    def update_exact(self, lay):
        self.A_exact[lay + 1] = func.contract_A_exact(self.A_list[lay + 1],type='16')
    
    def forward(self, lay):
        A_in = self.A_list[lay]
        A_new = self.layers_tnr[lay](A_in)
        A_exact = func.contract_A_exact(self.A_list[lay],type='4')
        
        return torch.abs(A_new - A_exact)

    def forward_trans(self, lay):
        A_in = self.A_list[lay]
        A_out = self.layers_tnr[lay].RG_step(A_in)
        # A_in = self.layers_div[lay + 1](A_out)
        #  * self.layers_div[lay + 1].norm ** 8
        A_new = self.layers_tnr[lay + 1](A_out)

        return torch.abs(A_new  - self.A_exact[lay]) / torch.abs(self.A_exact[lay])

    def sum_lnZ(self):
        r"""Compute the lnZ at the top layer.
        """
        lnZ = torch.log(torch.einsum('abab', self.A_list[-1]))
        i = 0
        for lay in reversed(self.layers_div):
            lnZ += 4 ** i * torch.log(lay.norm) 
            i += 1
        lnZ = lnZ / 4 ** (self.totlv) 

        return lnZ

    def sum_lnZ_trans(self):
        r"""Compute the lnZ at the top layer.
        """
        lnZ = torch.log(torch.einsum('abab', self.A_list[-1]))
        # i = 0
        for i in range(self.totlv, -1, -2):
            lnZ += 4 ** (self.totlv - i) * torch.log(self.layers_div[i].norm)

        lnZ = lnZ / 4 ** (self.totlv) 

        return lnZ

