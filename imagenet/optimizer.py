import torch
from torch.optim import Optimizer
import horovod.torch as hvd

import inspect

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


class SGDClipGrad(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    with clipped gradient.
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.

    As Pytorch's Distributed Data Parallel (DDP) starts from the same
    point across all nodes, and only use allreduce to broadcast the
    average gradient to be written to the param.grad field of all
    parameters. So after the backward pass, the grad field on the same
    corresponding parameter across different DDP processes should be
    the same.
    Given this, we can record the averaged gradient every I iterations.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, clipping_param=1e10):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clipping_param < 0.0:
            raise ValueError("Invalid clipping_param value: {}".format(clipping_param))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        clipping_param=clipping_param)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDClipGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDClipGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(
        self,
        closure=None,
        correction=None,
        local_correction=None,
        global_correction=None,
    ):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and
                returns the loss.
            correction (str, optional): Type of gradient correction (if any) to perform.
            local_correction (List[torch.Tensor], optional): Subtract from gradient for
                SCAFFOLD-style corrections.
            global_correction (List[torch.Tensor], optional): Add to gradient for
                SCAFFOLD-style corrections.
        """
        assert (local_correction is None) == (global_correction is None)
        assert local_correction is None or correction is not None
        assert len(self.param_groups) == 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute clipping coefficient and update.
        local_update_l2_norm_sq = 0.0
        i = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # Correct gradient, if necessary.
                if correction is not None and local_correction is not None:
                    d_p = d_p.add(global_correction[i] - local_correction[i])

                param_state['update'] = torch.clone(d_p).detach()
                local_update_l2_norm_sq += torch.sum(d_p.data * d_p.data)
                i += 1

        local_update_l2_norm = torch.sqrt(local_update_l2_norm_sq).item()
        if correction == "episode":
            global_update_l2_norm = torch.sqrt(
                torch.sum(torch.cat([g.view(-1) for g in global_correction]) ** 2)
            )
        clipping_coeff = group['clipping_param'] / (1e-10 + local_update_l2_norm)

        # Compute update size.
        indicator = global_update_l2_norm if correction == "episode" else local_update_l2_norm
        clip = indicator > group['clipping_param'] / group['lr']
        lr = clipping_coeff if clip else group['lr']

        # Perform update.
        for p in group['params']:
            if p.grad is None:
                continue
            param_state = self.state[p]
            p.add_(param_state['update'], alpha=-lr)

        return loss, int(clip)
