import torch
import numpy as np
from numpy import linalg as LA
from scipy import integrate

class NetTensor(object):
    r"""Constructing network tensor of various models.
    Args:
        model (str): name of models ('CIsing')
        **kwargs (dict, optional): Hamiltonian parameters of the model
    Attributes:
        E_exact (float): the exact value of g.s. energy for given parameters. 0 value for E_exact to be determined.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        ham_dict = {'CIsing':self.CIsing}
        ham_dict.get(self.model, self.default)(**self.kwargs)

    def default(self, **kwargs):
        raise Exception("The model \'" + self.model + "\' is not defined!")

    def CIsing(self, **kwargs):
        self.dtype = torch.double
        self.chi_HV = (2, 2)
        self.beta = torch.tensor(np.log(1 + np.sqrt(2)) / 2)
        if 'beta' in list(kwargs.keys()):
            self.beta = torch.tensor(kwargs['beta'])
        lam = [torch.cosh(self.beta)*2, torch.sinh(self.beta)*2]
        A = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        if ((i+j+k+l)%2==0):
                            A.append(torch.sqrt(lam[i]*lam[j]*lam[k]*lam[l])/2.)
                        else:
                            A.append(torch.tensor(0.0, dtype=self.dtype))
        A = torch.stack(A).view(2, 2, 2, 2)
        self.A = A
        
        self.beta0 = self.beta.cpu().detach().numpy()
        maglambda = 1 / (np.sinh(2 * self.beta0)** 2)
        Integrate, _ = integrate.quad(lambda x: np.log(np.cosh(2 * self.beta0)** 2 + np.sqrt(1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)) / maglambda), 0, np.pi)
        self.lnZ_exact = 0.5 * np.log(2) + Integrate / (2 * np.pi)

        K1,_ = integrate.quad(lambda x:1 / np.sqrt(1 - 4 * maglambda * np.sin(x)** 2 / (1 + maglambda + 1e-10)** 2), 0, np.pi/2)
        self.E_exact = -(K1 * 2 / np.pi * (2 * np.tanh(2 * self.beta0)** 2 - 1) + 1) / np.tanh(2 * self.beta0)
