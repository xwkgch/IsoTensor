import torch
import numpy as np
from numpy import linalg as LA
from scipy import integrate
import math

class Hamiltonian(object):
    r"""Constructing 1D Hamiltonian of various models.
    Args:
        model (str): name of models ('Ising', Heisenberg 'XY', Heisenberg 'XXZ', 'Potts')
        device (torch.device): device for computation 'cpu' or 'cuda'
        **kwargs (dict, optional): Hamiltonian parameters of the model
    Attributes:
        ham (tensor): two-site Hamiltonian tensor shifted by a bias.
        bias (float): the bias, which should be added when compute the energy.
        E_exact (float): the exact value of g.s. energy for given parameters. 0 value for E_exact to be determined.
    """
    def __init__(self, model, device, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.device = device
        ham_dict = {'Ising':self.Ising, 'Potts':self.Potts, 'XY':self.XY, 'XXZ':self.XXZ}
        ham_dict.get(self.model, self.default)(**self.kwargs)
        self.ham = self.ham.to(device)

    def default(self, **kwargs):
        raise Exception("The model \'" + self.model + "\' is not defined!")

    def Ising(self, **kwargs):
        self.g = 1.0
        self.name = 'Ising'
        self.description = ''
        self.dtype=torch.double
        self.g = torch.tensor(1.0, dtype=self.dtype)
        if 'g' in list(kwargs.keys()):
            self.g = kwargs['g']
        
        sX = torch.tensor([[0, 1.0], [1.0, 0]], dtype=self.dtype)
        sZ = torch.tensor([[1.0, 0], [0, -1.0]], dtype=self.dtype)
        H = torch.einsum('xz, wy -> zyxw', sX, sX) + 0.5 * self.g * (torch.einsum('xz, wy -> zyxw', sZ, torch.eye(2)) + torch.einsum('xz, wy -> zyxw', torch.eye(2), sZ))
        H = 0.5 * (H + H.permute(1, 0, 3, 2))
        self.bias = np.amax(torch.eig(H.reshape(4, 4))[0].view(-1).numpy())
        self.ham = H - self.bias * torch.eye(4).reshape(2, 2, 2, 2)

        integ, _ = integrate.quad(lambda x: np.sqrt((self.g - np.cos(x))**2 + np.sin(x)**2), -np.pi, np.pi) 
        self.E_exact = - integ / (2 * np.pi)

    def XY(self, **kwargs):
        self.g = 1.0
        if 'g' in list(kwargs.keys()):
            self.g = kwargs['g']
        self.name = 'Heisenberg XY'
        self.description = ''
        self.dtype=torch.double

        sX = torch.tensor([[0, 1.0], [1.0, 0]], dtype=self.dtype)
        sY = torch.tensor([[0, -1.0], [1.0, 0]], dtype=self.dtype)
        H = torch.einsum('xz, wy -> zyxw', sX, sX) - self.g * torch.einsum('xz, wy -> zyxw', sY, sY)
        H = 0.5 * (H + H.permute(1, 0, 3, 2))
        self.bias = np.amax(torch.eig(H.reshape(4, 4))[0].view(-1).numpy())
        self.ham = H - self.bias * torch.eye(4).reshape(2, 2, 2, 2)

        if self.g == 1.0:
            self.E_exact = -4 / math.pi
        else:
            self.E_exact = 0

    def XXZ(self, **kwargs):
        self.g = 1.0
        if 'g' in list(kwargs.keys()):
            self.g = kwargs['g']
        self.name = 'Heisenberg XXZ'
        self.description = ''
        self.dtype=torch.double

        sX = torch.tensor([[0, 1.0], [1.0, 0]], dtype=self.dtype)
        sY = torch.tensor([[0, -1.0], [1.0, 0]], dtype=self.dtype)
        sZ = torch.tensor([[1.0, 0], [0, -1.0]], dtype=self.dtype)
        H = torch.einsum('xz, wy -> zyxw', sX, sX) - torch.einsum('xz, wy -> zyxw', sY, sY) + self.g * torch.einsum('xz, wy -> zyxw', sZ, sZ)
        H = 0.5 * (H + H.permute(1, 0, 3, 2))
        self.bias = np.amax(torch.eig(H.reshape(4, 4))[0].view(-1).numpy())
        self.ham = H - self.bias * torch.eye(4).reshape(2, 2, 2, 2)

        if self.g == 1.0:
            self.E_exact = 1 - 4 * np.log(2)
        else:
            self.E_exact = 0

    def Potts(self, **kwargs):
        self.g = 1.0
        if 'g' in list(kwargs.keys()):
            self.g = kwargs['g']
        self.name = 'Potts'
        self.description = ''
        self.dtype=torch.double

        Mz = torch.tensor([[2, 0, 0], [0, -1.0, 0], [0, 0, -1.0]], dtype=self.dtype)
        Mx1 = torch.tensor([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]], dtype=self.dtype)
        Mx2 = torch.tensor([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]], dtype=self.dtype)
        H = torch.einsum('xz, wy -> zyxw', Mx1, Mx2) + torch.einsum('xz, wy -> zyxw', Mx2, Mx1) + 0.5 * self.g * (torch.einsum('xz, wy -> zyxw', Mz, torch.eye(3)) + torch.einsum('xz, wy -> zyxw', torch.eye(3), Mz))
        H = 0.5 * (H + H.permute(1, 0, 3, 2))
        self.bias = np.amax(torch.eig(H.reshape(9, 9))[0].view(-1).numpy())
        self.ham = H - self.bias * torch.eye(9).reshape(3, 3, 3, 3)

        # E_exact = -1.8156
        # l = (3 + 2 * math.sqrt(3) + 2) / (math.sqrt(3) + 2)
        # h = l / 3
        # E_exact = 1 - 0.5 * (l + h) - math.sqrt(0.25 * (l-h)**2 + 2*h**2)

        self.E_exact = 0
    
