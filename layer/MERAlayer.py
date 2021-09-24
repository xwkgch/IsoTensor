import sys
sys.path.append("..")
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import lib.functional as func

class SimpleLayer(torch.nn.Module):
    r"""Basic layer for MERA.
    Args:
        chi_in (int): dimension of in (bottom) bond
        chi_out (int): dimension of out (upper) bond
        dtype (tensor.dtype): data type (may be complex for some models)
    Attribute:
        u (Parameter): torch container packaging disentangler tensor u
        u.leg_type (tuple): a pair of int number indicating the number of in and out bonds of u ((2, 2) for u)
    """
    def __init__(self, chi_in, chi_out, dtype):
        assert chi_in > 0 and chi_out > 0
        super().__init__()
        self.chi_in = chi_in
        self.chi_out = chi_out
        self.dtype = dtype

        u = torch.eye(self.chi_in ** 2, dtype=self.dtype).view(self.chi_in, self.chi_in, self.chi_in, self.chi_in)
        self.u = Parameter(u)
        self.u.leg_type = (2, 2)


class SimpleTernary(SimpleLayer):
    r"""single layer for ternary MERA.
    Args:
        chi_in (int): dimension of in (bottom) bond
        chi_out (int): dimension of out (upper) bond
        leg_type (tuple): a pair of int number indicating the number of in and out bonds of w (default: (3, 1))
        dtype (tensor.dtype): data type (may be complex for some models)
    Attribute:
        u (Parameter): torch container packaging disentangler tensor u
        u.leg_type (tuple): a pair of int number indicating the number of in and out bonds ((2, 2) for u)
    """
    def __init__(self, chi_in, chi_out, leg_type, dtype):
        if not isinstance(leg_type, tuple) or not len(leg_type) == 2 or not leg_type[0] >= leg_type[1]:
            raise ValueError("w type should be a tuple with two elements: {}".format(leg_type))
        super().__init__(chi_in, chi_out, dtype)

        bond_in, bond_out = leg_type
        w = torch.randn(*((self.chi_in,)*bond_in + (self.chi_out,)*bond_out), dtype=self.dtype)
        w = func.normal(w, leg_type)
        self.w = Parameter(w)
        self.w.leg_type = leg_type

    def forward(self, rho):
        return func.des_ternary(rho, self.w, self.u)

    def padding(self, chi_in_new, chi_out_new):
        r"""expand the bond dimension of parameters and padding 0 on the new area
        """
        assert chi_in_new >= self.chi_in and chi_out_new >= self.chi_out

        d_in = chi_in_new - self.chi_in
        d_out = chi_out_new - self.chi_out
        bond_in, bond_out = self.w.leg_type
        self.u = Parameter(F.pad(self.u, (0, d_in)*4))
        self.u.leg_type = (2, 2)
        self.w = Parameter(F.pad(self.w, (0, d_out)*bond_out + (0, d_in)*bond_in))
        self.w.leg_type = (bond_in, bond_out)

        self.chi_in, self.chi_out = chi_in_new, chi_out_new
