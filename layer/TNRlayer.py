import sys
sys.path.append("..")
import torch
from torch.nn.parameter import Parameter
from lib.torchncon import ncon
import lib.functional as func


class LayerGen(torch.nn.Module):
    r"""Generating layer for 2D classical Ising model on square lattice.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, beta):
        return func.class_Ising(beta)

class LayerDiv(torch.nn.Module):
    r"""Dividing layer for tensor A.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x, self.norm = func.tensor_div(x)
        return x


class LayerTNR(torch.nn.Module):
    r"""Coarse-graining layer for TNR.
    """
    def __init__(self, chi_HV, chi_list, dtype):
        super().__init__()
        self.chi_HV = chi_HV
        chiHI, chiVI, chiU, chiV, chiAH, chiAV = func.get_chi(chi_HV, chi_list)

        self.chi_list = chi_list
        u = torch.einsum('ac, bd -> abcd', [torch.eye(chiVI, chiU, dtype=dtype), torch.eye(chiVI, chiU, dtype=dtype)])
        self.u = Parameter(u)
        v_tmp = torch.eye(chiHI * chiU, chiV, dtype=dtype).view(chiHI, chiU, chiV)
        self.vL = Parameter(func.normal(v_tmp, (2, 1)))
        self.vR = Parameter(func.normal(v_tmp, (2, 1)))
        self.u.leg_type = (2, 2)
        self.vL.leg_type = (2, 1)
        self.vR.leg_type = (2, 1)

    def RG_step(self, A):
        self.buf_A = A

        B = func.contract_B(A, self.u, self.vL, self.vR, mode='parallel')
        uB, vB = func.svd_refactor(B, self.chi_HV, self.chi_list)

        # with torch.no_grad():
        C2 = ncon([vB, uB, torch.conj(self.vR), torch.conj(self.vL), self.vR, self.vL], [[1, 4, -1], [2, 3, -4], [6, -2, 1], [6, -3, 2], [5, -5, 4], [5, -6, 3]], [4, 6, 5, 2, 1, 3])
        w = func.eig_opt(C2, self.chi_HV, self.chi_list)

        A_out = ncon([w,torch.conj(w),vB,uB,torch.conj(self.vR),torch.conj(self.vL),self.vR,self.vL],[[7,8,-2],[10,9,-4],[1,4,-1],[2,3,-3],[6,7,1],[6,8,2],[5,10,4],[5,9,3]],[7,5,10,9,3,8,6,1,2,4])

        return A_out
    
    def forward(self, A):
        A_new = func.contract_A_new(A, self.u, self.vL, self.vR)
        # A_exact = func.contract_A_exact(A_origin)

        # return torch.abs(A_new - A_exact)
        return A_new
        