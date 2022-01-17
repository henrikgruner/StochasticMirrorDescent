import torch
from torch.optim import Optimizer


class SMD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    .. math::

    """

    def __init__(self, params, lr, q=2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, q=q, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SMD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SMD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            q = group['q']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            for i, param_with_grad in enumerate(params_with_grad):

                d_p = d_p_list[i]
                if weight_decay != 0:
                    d_p = d_p.add(param_with_grad, alpha=weight_decay)

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

                grad = group['lr'] * d_p

                if(group['q'] == 1):
                    eps = 0.1
                    # instead of q = 1, q=1.01 to keep the function strictly convex and differentiable

                    # nabla psi(w_ij) = nabla psi(w_i-1,j) - lr * nabla L(w_i-1,j)
                    # nabla psi(w_ij) = |w_i-1,j|^(p-1)*sgn(w_i-1,j) - gamma*nabla L(w_i-1,j)

                    nabla_psi_prev_w = (
                        torch.abs(param_with_grad.data)**eps * torch.sign(param_with_grad.data))

                    nabla_psi_w = nabla_psi_prev_w - grad
                    param_with_grad.data = (
                        torch.abs(nabla_psi_w)**(1/eps)) * torch.sign(nabla_psi_w)

                else:
                    # nabla psi(w_ij) = nabla psi(w_i-1,j) - lr * nabla L(w_i-1,j)
                    # nabla psi(w_ij) = |w_i-1,j|^(p-1)*sgn(w_i-1,j) - gamma*nabla L(w_i-1,j)

                    nabla_psi_prev_w = torch.abs(param_with_grad.data)**(
                        group['q']-1) * torch.sign(param_with_grad.data)

                    nabla_psi_w = nabla_psi_prev_w - grad

                    param_with_grad.data = (
                        torch.abs(nabla_psi_w)**(1/(group['q'] - 1))) * torch.sign(nabla_psi_w)

        return loss
