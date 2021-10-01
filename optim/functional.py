import torch
from torch import Tensor
from typing import List, Optional
import numpy as np
from numpy import linalg as LA
import math

def grad_proj(X, dX, inner='canonical'):
    r"""project grad data onto Stiefel manifold
    Args:
        X (tensor): tensor of parameters in matrix form
        dX (tensor): grad tensor of parameters in matrix form
        inner (str, optional): inner product type on Stiefel manifold (default: 'canonical')
    Return:
        G (tensor): gradient tensor in matrix form
        A (tensor): the tensor G = AX in matrix form
    """
    G = dX - 0.5 * (X @ X.t().conj() @ dX + X @ dX.t().conj() @ X)
    A = dX @ X.t().conj() - X @ dX.t().conj() + 0.5 * X @ (dX.t().conj() @ X - X.t().conj() @ dX) @ X.t().conj()
    return G, A
    

def retraction(X, G, A, lr, method='SVD', adapt=False):
    r"""retract the updated tensor onto Stiefel manifold
    Args:
        X (tensor): tensor of parameters in matrix form
        G (tensor): gradient tensor in matrix form
        A (tensor): the tensor G = AX in matrix form
        lr (float): learning rate
        method (str, optional): method for retraction into Stiefel manifold ('SVD', 'Cayley') (default: 'SVD')
    """
    if adapt:
        lr = min(lr, 4.0 / (torch.linalg.norm(A).item() + 1e-8)) 
    if method == 'SVD':
        U, _, Vt = LA.svd((X - lr * G).cpu().numpy(), full_matrices=False)
        X_out = torch.tensor(U @ Vt)
        # U, _, V = svd_(X - lr * G)
        # X_out = U @ V.t()

    elif method == 'Cayley':
        X_out = X - lr * G
        for i in range(40):
            X_out = X - 0.3 * lr * A @ (X + X_out)

    return X_out

def evenbly_vidal(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float):
    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        dims = param.size()
        bond_in, _ = param.leg_type
        dim1 = np.prod(dims[:bond_in])
        dim2 = np.prod(dims[bond_in:])
        X = param.data.view(dim1, dim2)
        dX = d_p.data.view(dim1, dim2)

        U, _, Vt = LA.svd(dX.cpu().numpy(), full_matrices=False)
        X_out = torch.tensor(U @ Vt)
        param.data.fill_(0)
        param.data += X_out.view(dims).to(param.device)

    
def iso_sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        method: str):
    r"""Functional API that performs SGD algorithm computation under isometric constraints.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        dims = param.size()
        bond_in, _ = param.leg_type
        dim1 = np.prod(dims[:bond_in])
        dim2 = np.prod(dims[bond_in:])
        X = param.data.view(dim1, dim2)
        dX = d_p.data.view(dim1, dim2)

        G, A = grad_proj(X, dX)
        momentum_buffer_list[i] = torch.clone(G.view(dims)).detach()
        X_out = retraction(X, G, A, lr, method=method, adapt=True)
        param.data.fill_(0)
        param.data += X_out.view(dims).to(param.device)


def iso_adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         method:str):
    r"""Functional API that performs Adam algorithm computation.

    """
    keep_dir = True

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        #param.addcdiv_(exp_avg, denom, value=-step_size)

        d_p = exp_avg

        dims = param.size()
        bond_in, _ = param.leg_type
        dim1 = np.prod(dims[:bond_in])
        dim2 = np.prod(dims[bond_in:])
        X = param.data.view(dim1, dim2)
        dX = d_p.data.view(dim1, dim2)

        G, A = grad_proj(X, dX)
        exp_avgs[i] = torch.clone(G.view(dims)).detach()
        
        r = torch.linalg.norm(denom)
        if keep_dir:
            G = G / r * 20
        else:
            G = torch.div(G, denom.view(dim1, dim2)) 
            G, A = grad_proj(X, G)

        X_out = retraction(X, G, A / r, step_size, method=method, adapt=False)
        param.data.fill_(0)
        param.data += X_out.view(dims).to(param.device)

def rmsprop(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            grad_avgs: List[Tensor],
            momentum_buffer_list: List[Tensor],
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool,
            method:str):
    r"""Functional API that performs rmsprop algorithm computation.
    """
    keep_dir = True

    for i, param in enumerate(params):
        grad = grads[i]
        square_avg = square_avgs[i]
        buf = momentum_buffer_list[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-0.5).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        # buf.mul_(momentum).addcdiv_(grad, avg)
        # param.add_(buf, alpha=-lr)


        r = torch.linalg.norm(avg)
        if keep_dir:
            buf = momentum * buf + grad / r 
        else: 
            buf = momentum * buf + grad

        d_p = buf

        dims = param.size()
        bond_in, _ = param.leg_type
        dim1 = np.prod(dims[:bond_in])
        dim2 = np.prod(dims[bond_in:])
        X = param.data.view(dim1, dim2)
        dX = d_p.data.view(dim1, dim2)

        G, A = grad_proj(X, dX)
        momentum_buffer_list[i] = torch.clone(G.view(dims)).detach()

        if not keep_dir:
            G = torch.div(G, avg.view(dim1, dim2)) / 4
            G, A = grad_proj(X, G)

        X_out = retraction(X, G, A / r, lr, method=method, adapt=False)
        param.data.fill_(0)
        param.data += X_out.view(dims).to(param.device)