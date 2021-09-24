import torch
from torch.optim import Optimizer
from . import functional as func

class EV(Optimizer):
    r"""Implements Evenbly-Vidal optimization.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.label = 'Evenbly-Vidal           '
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(EV, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(EV, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            weight_decay = group['weight_decay']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            func.evenbly_vidal(params_with_grad,
                  d_p_list,
                  weight_decay)

        return loss