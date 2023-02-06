from math import sqrt

import torch
from torch.optim import Optimizer


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

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 clipping_param=0, algorithm='single_clip'):
        if isinstance(params, dict):
            raise ValueError("Only a single param group is supported.")
        if lr and lr < 0.0:
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

        self.algorithm = algorithm

    def __setstate__(self, state):
        super(SGDClipGrad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(
        self,
        global_average_grad_l2_norm,
        local_avg_grad,
        global_avg_grad,
        local_control_var=None,
        global_control_var=None,
        local_clip_l2_norm=None,
        global_clip_l2_norm=None,
        first=None,
        closure=None
    ):
        """Performs a single optimization step.
        Arguments:
            global_average_grad_l2_norm (float): the l2 norm of the averaged gradient.
            local_avg_grad (List[torch.Tensor]): the average local gradient across the
                previous communication round. Only used for SCAFFOLD.
            global_avg_grad (List[torch.Tensor]): the average global gradient across the
                previous communication round. Only used for SCAFFOLD.
            local_control_var (List[torch.Tensor]): the average local gradient as
                sampled at the beginning of the current communication round. Only used
                for corrected_clip.
            global_control_var (List[torch.Tensor]): the average global gradient as
                sampled at the beginning of the current communication round. Only used
                for corrected_clip.
            local_clip_l2_norm (float): L2 norm of local gradient sampled to determine
                clipping decision. Only used for corrected_clip and EPISODE variants.
            global_clip_l2_norm (float): L2 norm of global gradient sampled to determine
                clipping decision. Only used for episode_double.
            first (bool): Whether or not this step is the first in its communication
                round. Only used for episode_double.
            closure (callable, optional): A closure that reevaluates the model and
                returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Compute perform_clip for EPISODE and variants. We do this here so that we know
        # whether to compute the L2 norm of the gradient or the L2 norm of the corrected
        # gradient for episode_scaffold.
        perform_clip = None
        perform_correction = None
        if self.algorithm in ["corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "scaffold_clip", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"]:
            assert len(self.param_groups) == 1
            group = self.param_groups[0]
            if self.algorithm == "corrected_clip":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = not perform_clip
            elif self.algorithm == "episode_scaffold":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            elif self.algorithm == "episode_normal_1":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            elif self.algorithm == "episode_double":
                perform_clip = (global_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = (local_clip_l2_norm <= group['clipping_param'] / group['lr'])
            elif self.algorithm == "episode_practical":
                perform_clip = (global_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            elif self.algorithm == "scaffold_clip":
                perform_correction = True
            elif self.algorithm == "episode_inverted":
                perform_clip = (global_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = perform_clip
            elif self.algorithm == "episode_final":
                perform_clip = (global_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            elif self.algorithm == "delayed_clip":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = False
            elif self.algorithm == "test_correction":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            elif self.algorithm == "episode_balanced":
                perform_clip = (local_clip_l2_norm >= group['clipping_param'] / group['lr'])
                perform_correction = True
            else:
                raise NotImplementedError

        # If using episode_double, not correcting, and it isn't the first step of a
        # communication round, do not perform any update.
        if self.algorithm == "episode_double":
            assert first is not None
            if not perform_correction and not first:
                return loss, None, None

        # Compute norm of control variates.
        global_norm = None
        if self.algorithm in ["episode_normal_1", "episode_balanced"]:
            global_norm = 0
            for i in range(len(global_control_var)):
                if global_control_var[i] is not None:
                    global_norm += float(torch.sum(global_control_var[i] * global_control_var[i]))
            global_norm = sqrt(global_norm)
        local_norm = None
        if self.algorithm == "episode_balanced":
            local_norm = 0
            for i in range(len(local_control_var)):
                if local_control_var[i] is not None:
                    local_norm += float(torch.sum(local_control_var[i] * local_control_var[i]))
            local_norm = sqrt(local_norm)
        grad_norm = None
        if self.algorithm == "episode_balanced":
            grad_norm = 0
            assert len(self.param_groups) == 1
            group = self.param_groups[0]
            for i, p in enumerate(group['params']):
                if p.grad is not None:
                    grad_norm += float(torch.sum(p.grad.data * p.grad.data))
            grad_norm = sqrt(grad_norm)

        # Compute clipping coefficient
        local_grad_l2_norm = 0.0
        if self.algorithm in ["single_clip", "local_clip", "max_clip", "corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "scaffold_clip", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"]:
            local_grad_l2_norm_sq = 0.0
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue

                    if self.algorithm in ["corrected_clip", "episode_scaffold", "episode_inverted", "episode_final", "test_correction"]:
                        if perform_correction:
                            corrected_grad = p.grad + global_control_var[i] - local_control_var[i]
                            local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                        else:
                            local_grad_l2_norm_sq += torch.sum(p.grad.data * p.grad.data)
                    elif self.algorithm == "episode_normal_1":
                        if perform_correction:
                            corrected_grad = p.grad + global_control_var[i] / global_norm * group["clipping_param"] / group["lr"] - local_control_var[i]
                            local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                        else:
                            local_grad_l2_norm_sq += torch.sum(p.grad.data * p.grad.data)
                    elif self.algorithm == "episode_double":
                        if perform_correction:
                            corrected_grad = p.grad + global_control_var[i] - local_control_var[i]
                        else:
                            corrected_grad = global_control_var[i]
                        local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                    elif self.algorithm == "episode_practical":
                        if perform_correction and local_avg_grad is not None and global_avg_grad is not None:
                            corrected_grad = p.grad + global_avg_grad[i] - local_avg_grad[i]
                            local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                        else:
                            local_grad_l2_norm_sq += torch.sum(p.grad.data * p.grad.data)
                    elif self.algorithm == "scaffold_clip":
                        if local_avg_grad is not None and global_avg_grad is not None:
                            corrected_grad = p.grad + global_avg_grad[i] - local_avg_grad[i]
                            local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                        else:
                            local_grad_l2_norm_sq += torch.sum(p.grad.data * p.grad.data)
                    elif self.algorithm == "episode_balanced":
                        n_grad = p.grad / grad_norm
                        n_c_i = local_control_var[i] / local_norm
                        n_c = global_control_var[i] / global_norm
                        corrected_grad = n_grad + n_c - n_c_i
                        local_grad_l2_norm_sq += torch.sum(corrected_grad.data * corrected_grad.data)
                    else:
                        local_grad_l2_norm_sq += torch.sum(p.grad.data * p.grad.data)

            local_grad_l2_norm = torch.sqrt(local_grad_l2_norm_sq).item()

        # Test accuracy of gradient correction.
        if self.algorithm == "test_correction" and local_avg_grad is not None:

            assert len(self.param_groups) == 1
            group = self.param_groups[0]

            grad = torch.cat([p.grad.view(-1) for p in group['params'] if p.grad is not None])
            c_i = torch.cat([g.view(-1) for g in local_avg_grad if g is not None])
            c = torch.cat([g.view(-1) for g in global_avg_grad if g is not None])
            G_ri = torch.cat([g.view(-1) for g in local_control_var if g is not None])
            G_r = torch.cat([g.view(-1) for g in global_control_var if g is not None])
            scaffold_correction = grad - c_i + c
            episode_correction = grad - G_ri + G_r

            # Compare local gradient with corrected gradient.
            norm = lambda x: float(torch.sqrt(torch.sum(x ** 2)))
            ang = lambda x, y: float(torch.sum(x * y)) / (norm(x) * norm(y))
            vec_diff = lambda x, y: (norm(x - y), ang(x, y))
            grad_local = vec_diff(grad, G_ri)
            grad_total = vec_diff(grad, G_r)
            scaffold_local = vec_diff(grad, c_i)
            scaffold_total = vec_diff(scaffold_correction, G_r)
            episode_local = vec_diff(grad, G_ri)
            episode_total = vec_diff(episode_correction, G_r)

            # Store comparison.
            if not hasattr(self, "grad_local_diffs"):
                self.grad_local_diffs = []
                self.grad_total_diffs = []
                self.scaffold_local_diffs = []
                self.scaffold_total_diffs = []
                self.episode_local_diffs = []
                self.episode_total_diffs = []
            self.grad_local_diffs.append(grad_local)
            self.grad_total_diffs.append(grad_total)
            self.scaffold_local_diffs.append(scaffold_local)
            self.scaffold_total_diffs.append(scaffold_total)
            self.episode_local_diffs.append(episode_local)
            self.episode_total_diffs.append(episode_total)

        total_clip_computations = 0
        clipped_computations = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad

                if self.algorithm in ["SCAFFOLD", "scaffold_clip"]:
                    if local_avg_grad is not None and global_avg_grad is not None:
                        d_p += global_avg_grad[i] - local_avg_grad[i]
                elif self.algorithm in ["corrected_clip", "episode_scaffold", "episode_inverted", "episode_final", "test_correction"]:
                    if perform_correction:
                        d_p += global_control_var[i] - local_control_var[i]
                elif self.algorithm == "episode_normal_1":
                    if perform_correction:
                        d_p += global_control_var[i] / global_norm * group["clipping_param"] / group["lr"] - local_control_var[i]
                elif self.algorithm == "episode_double":
                    if perform_correction:
                        d_p += global_control_var[i] - local_control_var[i]
                    else:
                        d_p = global_control_var[i]
                elif self.algorithm == "episode_practical":
                    if perform_correction and local_avg_grad is not None and global_avg_grad is not None:
                        d_p += global_avg_grad[i] - local_avg_grad[i]
                elif self.algorithm == "episode_balanced":
                    d_p /= grad_norm
                    d_p += global_control_var[i] / global_norm - local_control_var[i] / local_norm

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                clipping_coeff = None
                if self.algorithm == 'max_clip':
                    clipping_coeff = group['clipping_param'] / (1e-10 + max(local_grad_l2_norm, global_average_grad_l2_norm))
                elif self.algorithm in ['single_clip', 'local_clip', 'corrected_clip', 'episode_scaffold', 'episode_normal_1', 'episode_double', 'episode_practical', 'scaffold_clip', 'episode_inverted', 'episode_final', 'delayed_clip', 'test_correction', 'episode_balanced']:
                    clipping_coeff = group['clipping_param'] / (1e-10 + local_grad_l2_norm)
                elif self.algorithm == 'global_avg_clip':
                    clipping_coeff = group['clipping_param'] / (1e-10 + global_average_grad_l2_norm)

                if clipping_coeff is not None:
                    if perform_clip is None:
                        lr = min(group['lr'], clipping_coeff)
                        clipped_computations += 1 if clipping_coeff < group['lr'] else 0
                    else:
                        if perform_clip:
                            lr = clipping_coeff
                            clipped_computations += 1
                        else:
                            lr = group['lr']
                    total_clip_computations += 1
                else:
                    lr = group["lr"]

                p.add_(d_p, alpha=-lr)

        corrected_computations = 1 if perform_correction else 0
        return loss, clipped_computations / (total_clip_computations + 1e-10), corrected_computations
