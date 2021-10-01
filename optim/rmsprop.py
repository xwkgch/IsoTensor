import torch
from torch.optim import Optimizer
from . import functional as func


class RMSprop(Optimizer):
    r"""Implements RMSprop algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, method='SVD'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        self.label = 'RMSprop with ' + method + '       '
        self.pad_item = ['square_avg', 'momentum_buffer']
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay, method=method)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

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
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if not 'step' in state.keys():
                    state['step'] = self._step_count - 1
                if not 'square_avg' in state.keys():
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if not 'momentum_buffer' in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group['centered']:
                    if not 'grad_avg' in self.pad_item:
                        self.pad_item.append('grad_avg')
                    if not 'grad_avg' in state.keys():
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1


            func.rmsprop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      group['lr'],
                      group['alpha'],
                      group['eps'],
                      group['weight_decay'],
                      group['momentum'],
                      group['centered'],
                      group['method'])
            
            for p, buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                # state['square_avg'] = square_avg
                state['momentum_buffer'] = buffer

        return loss
