import torch
import torch.nn.functional as F
from .torchncon import ncon
from .ncon import ncon as ncon0
import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from .svd import SVD
from .eigh import EigenSolver
svd_ = SVD.apply
eigh_ = EigenSolver.apply

def normal(tensor, type):
    r"""renormalize a random tensor to be (semi-) unitary
    """
    dims = tensor.size()
    bond_in, _ = type
    dim1 = np.prod(dims[:bond_in])
    dim2 = np.prod(dims[bond_in:])

    tensor = tensor.view(dim1, dim2)
    tensor, _, _ = torch.svd(tensor)
    return tensor.view(dims)

def des_ternary(rho, w, u):
    r"""descending operation for ternary MERA
    """
    wt = torch.conj(w).detach()
    ut = torch.conj(u).detach()
    legs1 = [[3, 4, 1, 2], [9, -3, 7, 1], [8, 11, 10, 2], [9, -1, 5, 3], [6, 11, 10, 4], [-4, 12, 7, 8], [-2, 12, 5, 6]]
    legs2 = [[3, 4, 1, 2], [9, 11, 7, 1], [8, 12, 10, 2], [9, 11, 5, 3], [6, 12, 10, 4], [-3, -4, 7, 8], [-1, -2, 5, 6]]
    legs3 = [[3, 4, 1, 2], [9, 11, 7, 1], [8, -4, 10, 2], [9, 11, 5, 3], [6, -2, 10, 4], [12, -3, 7, 8], [12, -1, 5, 6]]
    order1 = [11, 10, 4, 2, 8, 1, 7, 6, 12, 3, 9, 5]
    order2 = [9, 11, 3, 1, 12, 10, 4, 2, 5, 6, 7, 8]
    order3 = [9, 11, 3, 1, 4, 5, 6, 7, 12, 2, 10, 8]
    tensors = [rho, w, w, wt, wt, u, ut]
    
    rho_out = (ncon(tensors, legs1, order1) + ncon(tensors, legs2, order2) + ncon(tensors, legs3, order3)) / 3
    
    return rho_out

def topdense(rho, w, u, des_func, sciter=4):
    r"""update density tensor of top layer (as inputs of network) by iteration
    """
    for _ in range(sciter):
        rho_tmp = des_func(rho, w, u) 
        rho = 0.5 * (rho_tmp + torch.conj(rho_tmp.permute(2, 3, 0, 1))) / torch.einsum('abab', rho_tmp)
    
    return rho

def rho_init(chi, dtype=torch.double, rho=None, type='two'):
    r"""create or expand density tensor of top layer
    Args:
        chi (int): bond dimension of density tensor
        dtype (tensor.dtype): data type of density tensor (no need for tensor expanding)
        rho (tensor): density tensor for iteration (no need for tensor creating)
        type (str): TBD
    """
    if rho == None:
        return torch.eye(chi ** 2, dtype=dtype).view(chi, chi, chi, chi)
    else:
        d_in = chi - rho.size(0)
        assert d_in >= 0
        return F.pad(rho, (0, d_in)*4)

def pad(tensor, dims_new):
    r"""Expand the dimensions of the tensor and padding 0 on the new area.
    Args:
        tensor (torch.Tensor): tensor to be padded.
        dims_new: dimensions of the padded tensor.
    """
    dims = np.array(tensor.size())
    dims_new = np.array(dims_new)
    assert len(dims) == len(dims_new)
    d = dims_new - dims
    d = d[::-1]
    pad_list = []
    for i in range(len(d)):
        pad_list.append(0)
        pad_list.append(d[i])
    
    return F.pad(tensor, tuple(pad_list))
#-------------------------------------------------------------------------
def class_Ising(beta):
    r"""Generate the 2D classical Ising tensor network on square lattice with inverse temperature beta.
    """
    lam = [torch.cosh(beta)*2, torch.sinh(beta)*2]
    A = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if ((i+j+k+l)%2==0):
                        A.append(torch.sqrt(lam[i]*lam[j]*lam[k]*lam[l])/2.)
                    else:
                        A.append(torch.tensor(0.0, dtype=beta.dtype, device=beta.device))
    A = torch.stack(A).view(2, 2, 2, 2)

    return A

def get_chi(chi_HV, chi_list):
    r"""Compute the bond dimensions used the horizon and vertical dimensions of the input tensor A.
    Args:
        chi_HV (tuple): a tuple with two elements indicating the horizon and vertical dimensions of A.
        chi_list (list): a list of max bond dimensions [chiU, chiV, chiAH, chiAV]
    Return:
        chiHI, chiVI, chiU, chiV, chiAH, chiAV
    """
    with torch.no_grad():
        chiHI, chiVI = chi_HV
        chiU = min(chi_list[0], chiVI)
        chiV = min(chi_list[1], chiHI * chiU)
        chiAH = min(chi_list[2], chiV ** 2)
        chiAV = min(chi_list[3], chiU ** 2)

    return chiHI, chiVI, chiU, chiV, chiAH, chiAV

def tensor_div(A, factor='norm'):
    r"""divide the tensor A by a constant (the norm of A).
    """
    with torch.no_grad():
        norm = torch.linalg.norm(A)

    A_out = A / norm

    return A_out, norm

def svd_refactor(B, chi_HV, chi_list):
    r"""Implement SVD for tensor B and absorb the diagonal matrix.
    Return:
        a tuple (uB, vB)
    """
    chiHI, chiVI, chiU, chiV, chiAH, chiAV = get_chi(chi_HV, chi_list)
    uB, sB, vB = svd_(B.reshape(chiV ** 2, chiV ** 2))
    vB = torch.conj(vB.t().conj())
    if B.dtype == torch.cdouble:
        with torch.no_grad():
            sB = torch.stack([sB, torch.zeros(sB.shape[0], device=B.device)])
            sB = torch.view_as_complex(sB.t().conj().contiguous())

    uB = (uB[:,:chiAH] @ torch.diag(torch.sqrt(sB[:chiAH]))).reshape(chiV, chiV, chiAH)
    vB = (torch.diag(torch.sqrt(sB[:chiAH])) @ vB[:chiAH,:]).reshape(chiAH, chiV, chiV).permute(1, 2, 0)

    return uB, vB

def eig_opt(C, chi_HV, chi_list):
    r"""Implement eigen decomposition for hermite tensor C.
    Return:
        w
    """
    chiHI, chiVI, chiU, chiV, chiAH, chiAV = get_chi(chi_HV, chi_list)
    # _, w_tmp = torch.linalg.eigh(ncon([C, torch.conj(C)], [[1, -1, -2, 2, 3, 4], [1, -3, -4, 2, 3, 4]]).reshape(chiU ** 2, chiU ** 2))
    _, w_tmp = eigh_(ncon([C, torch.conj(C)], [[1, -1, -2, 2, 3, 4], [1, -3, -4, 2, 3, 4]]).reshape(chiU ** 2, chiU ** 2))
    w = w_tmp.reshape(chiU, chiU, w_tmp.shape[1])[:,:, range(-1, -chiAV - 1, -1)]
     
    return w

def contract_B(A, u, vL, vR, mode='parallel'):
    r"""Compute the B by tensor contraction.
    mode = 'parallel' for A with parallel arrangement and 'mirror' for A with mirror arrangement.
    Return:
        B
    """
    if mode == 'parallel':
        B = ncon([A,A,torch.conj(A),torch.conj(A),u,torch.conj(u),vR,vL,torch.conj(vR),torch.conj(vL)],[[5,1,13,6],[13,2,15,7],[12,8,14,6],[14,9,16,7],[1,2,3,4],[8,9,10,11],[15,4,-3],[5,3,-1],[16,11,-4],[12,10,-2]],[15,2,4,5,16,9,11,8,14,12,10,1,13,3,6,7])
    elif mode == 'mirror':
        B = ncon([A,A,torch.conj(A),torch.conj(A),u,torch.conj(u),vR,vL,torch.conj(vR),torch.conj(vL)],[[5,1,13,6],[15,2,13,7],[12,8,16,6],[14,9,16,7],[1,2,3,4],[8,9,10,11],[15,4,-3],[5,3,-1],[14,11,-4],[12,10,-2]],[15,2,4,5,14,9,11,8,16,12,10,1,13,3,6,7])
    return B

def contract_A_exact(A, mode='parallel', type='4'):
    r"""Compute the the exact tensor contraction of sub-graph with A.
    mode = 'parallel' for A with parallel arrangement and 'mirror' for A with mirror arrangement.
    """
    if type == '4':
        tensors = [A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A)]
        if mode == 'parallel':
            connects = [[13,15,3,1],[3,16,14,2],[11,9,4,1],[4,10,12,2],[13,15,7,5],[7,16,14,6],[11,9,8,5],[8,10,12,6]]
            con_order = [10,12,11,9,16,14,2,6,4,8,7,5,13,15,3,1]
        elif mode == 'mirror':
            connects = [[8,9,13,1],[11,12,13,2],[7,5,14,1],[10,6,14,2],[8,9,16,3],[11,12,16,4],[7,5,15,3],[10,6,15,4]]
            con_order = [11,12,10,6,2,4,7,5,14,15,16,3,8,9,13,1]

    if type == '16':
        tensors = [A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A),A,A,torch.conj(A),torch.conj(A)]
        connects = [[59,61,1,2],[1,62,41,3],[57,4,5,2],[5,6,42,3],[53,47,7,8],[7,48,44,9],[55,4,10,8],[10,6,43,9],[41,63,11,12],[11,64,60,13],[42,14,15,12],[15,16,58,13],[44,49,17,18],[17,50,54,19],[43,14,20,18],[20,16,56,19],[53,47,21,22],[21,48,45,23],[55,24,25,22],[25,26,46,23],[59,61,27,28],[27,62,52,29],[57,24,30,28],[30,26,51,29],[45,49,31,32],[31,50,54,33],[46,34,35,32],[35,36,56,33],[52,63,37,38],[37,64,60,39],[51,34,40,38],[40,36,58,39]]
        con_order = [53,47,59,61,50,54,64,60,1,27,62,5,7,21,48,10,11,63,37,15,17,49,31,20,25,30,35,40,2,3,28,29,57,8,9,22,23,55,12,13,39,38,58,41,52,42,51,18,19,33,32,56,14,16,34,36,4,6,24,26,44,45,43,46]
    

    return ncon(tensors,connects,con_order)

def contract_A_new(A, u, vL, vR, mode='parallel'):
    r"""Compute the the approximate tensor contraction of sub-graph with A.
    mode = 'parallel' for A with parallel arrangement and 'mirror' for A with mirror arrangement.
    """
    B = contract_B(A, u, vL, vR, mode=mode)
    tensors = [torch.conj(vR),torch.conj(vL),torch.conj(u),vR,vL,u,A,A,torch.conj(A),torch.conj(A),B]
    if mode == 'parallel':
        connects = [[12,6,1],[9,5,2],[10,11,5,6],[13,8,3],[16,7,4],[15,14,7,8],[9,10,19,17],[19,11,12,18],[16,15,20,17],[20,14,13,18],[2,4,1,3]]
        con_order = [6,12,11,10,19,5,9,1,2,3,18,13,8,14,17,20,15,4,7,16]
    elif mode == 'mirror':
        connects = [[19,6,1],[9,5,2],[10,20,5,6],[16,8,3],[13,7,4],[12,11,7,8],[9,10,18,14],[19,20,18,15],[13,12,17,14],[16,11,17,15],[2,4,1,3]]
        con_order = [6,19,20,10,18,5,9,1,2,3,15,16,8,11,14,17,12,4,7,13]

    return ncon(tensors,connects,con_order)


#------------------------------------------------------------------------

def mera_sc(w, cut=6):
    r"""Extract the scaling dimensions from the top isometry w in MERA.
    Args:
        cut (int): A integer number making a cut on the scaling dimensions.
    return:
        A list of scaling dimensions.
    """
    with torch.no_grad():
        chi = w.size(-1)
        op = torch.einsum('awby, axbz -> zyxw', [w, torch.conj(w)])
        op = op.reshape(chi ** 2, chi ** 2)
        op = op.cpu()
        op = op.numpy()
        dtemp, _ = eigs(op, k=cut, which='LM')
        scDims = - np.log(abs(dtemp)) / np.log(3)

    return scDims

def tnr_sc(A, N):
    r"""Extract the scaling dimensions from the top tensor A in TNR.
    Args:
        N is the number of sites
    return:
        A list of scaling dimensions.
    """
    # The memory cost is chi^N. For a 16GB RAM, keep chi^N<=2^20
    def TM_sparse(A, N, psi):
        # Input A: rank 4 tensor with chi*chi*chi*chi
        # N: the number of sites
        # psi: Input state with dimension chi^N
        chi = A.shape[0]
        psi_mat = psi.reshape([chi, chi**(N-1)])
        Apsi = ncon0([A,psi_mat],[[-1,-2,-3,1],[1,-4]])
        for j in range(N-1):
            Apsi_mat = Apsi.reshape([chi,chi**(j+1),chi,chi,chi**(N-j-2)])
            Apsi = ncon0([A,Apsi_mat],[[1,-3,-4,2],[-1,-2,1,2,-5]])
        Apsi = ncon0([Apsi],[[1,-1,-2,1,-3]])
        Apsi = Apsi.reshape([chi**N])
        return Apsi
    A = A.cpu().detach().numpy()
    chi = A.shape[0]
    TM = lambda psi:TM_sparse(A,N,psi)
    Es, _ = eigs(LinearOperator((chi**N,chi**N),matvec=TM), k = 50)
    ds = -np.log(np.real(Es))
    ds = (ds-ds[0])/(ds[2]-ds[0])
    ds = np.sort(ds)
    return ds