"""
Train a model on the training set.
"""
import copy
import time
import math

import numpy as np
import torch
import torch.distributed as dist

from evaluate import evaluate
from sgd_clip import SGDClipGrad


# Model Averaging
def average_model(world_size, model, group):
    for param in model.parameters():
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param.data /= world_size
        dist.broadcast(param.data, src=0, group=group)

# Gradient Averaging
def average_grad(world_size, model, group):
    for param in model.parameters():
        dist.reduce(param.grad.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param.grad.data /= world_size
        dist.broadcast(param.grad.data, src=0, group=group)

# Average dictionary of parameters
def average_list(world_size, l, group):
    for param in l:
        if param is None:
            continue
        dist.reduce(param, dst=0, op=dist.ReduceOp.SUM, group=group)
        param /= world_size
        dist.broadcast(param, src=0, group=group)

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, train_loader, test_loader, extra_loaders, net, criterion):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
        extra_loaders: A list of iterators over the training set, which enables gradient
            samples independent from those drawn for training. Only used for algorithm
            'corrected_clip'.
        net: the neural network model employed.
        criterion: the loss function.

    Outputs:
        All training losses, training accuracies, test losses, and test
        accuracies on each evaluation during training.
    """
    optimizer = SGDClipGrad(params=net.parameters(), lr=args.eta0, momentum=args.momentum,
                            weight_decay=args.weight_decay, nesterov=args.nesterov,
                            clipping_param=args.clipping_param, algorithm=args.algorithm)

    world_size = args.world_size
    group = dist.new_group(range(world_size))
    net_clone = copy.deepcopy(net)

    global_average_grad_l2_norm = 0.0
    prev_local_avg_grad = None
    prev_global_avg_grad = None
    local_avg_grad = None
    global_avg_grad = None
    local_control_var = None
    global_control_var = None
    local_clip_l2_norm = None
    global_clip_l2_norm = None

    for loader in extra_loaders:
        loader.reset()
    eval_steps = [
        int((i + 1) * (train_loader.num_batches / args.evals_per_epoch)) - 1
        for i in range(args.evals_per_epoch)
    ]

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    eval_elasped_times = []
    eval_clip_operations = []
    clip_operations = []
    eval_corrected_operations = []
    corrected_operations = []
    grad_diffs = []
    local_grads = []
    avg_grads = []
    t_total = 0
    eval_start = time.time()
    for epoch_idx in range(1, args.train_epochs + 1):
        net.train()

        train_loader.reset()
        t = 0
        while train_loader.has_next():
            data = train_loader.next_batch()

            # print(f'Rank {args.rank} -- {t_total}')
            if 0 == t_total % args.communication_interval:
                with torch.no_grad():
                    average_model(world_size, net, group)

                if args.algorithm in ["SCAFFOLD", "episode_practical", "scaffold_clip", "test_correction"] and local_avg_grad is not None:
                    global_avg_grad = []
                    for param_grad in local_avg_grad:
                        if param_grad is None:
                            global_avg_grad.append(None)
                        else:
                            global_avg_grad.append(param_grad.clone())
                    average_list(world_size, global_avg_grad, group)

                    # Reset SCAFFOLD statistics.
                    prev_local_avg_grad = []
                    prev_global_avg_grad = []
                    for param_grad in local_avg_grad:
                        if param_grad is not None:
                            prev_local_avg_grad.append(param_grad.clone())
                        else:
                            prev_local_avg_grad.append(None)
                    for param_grad in global_avg_grad:
                        if param_grad is not None:
                            prev_global_avg_grad.append(param_grad.clone())
                        else:
                            prev_global_avg_grad.append(None)
                    local_avg_grad = None
                    global_avg_grad = None

                if args.algorithm in [
                    "corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"
                ]:
                    # Sample gradient to compute control variates for current client.
                    if args.algorithm not in ["episode_practical", "episode_inverted", "episode_final", "delayed_final", "test_correction", "episode_balanced"]:
                        if not extra_loaders[0].has_next():
                            extra_loaders[0].reset()
                        data = extra_loaders[0].next_batch()
                        (s1_embed, s2_embed), (s1_lens, s2_lens), targets = data
                        s1_embed, s2_embed = s1_embed.cuda(), s2_embed.cuda()
                        targets = targets.cuda()
                        net.zero_grad()
                        scores = net((s1_embed, s1_lens), (s2_embed, s2_lens))
                        loss = criterion(scores, targets)
                        loss.backward()
                        local_control_var = []
                        for param in net.parameters():
                            if param.grad is None:
                                local_control_var.append(None)
                            else:
                                local_control_var.append(param.grad.clone())

                        # Average control variates across clients.
                        global_control_var = []
                        for grad in local_control_var:
                            if grad is None:
                                global_control_var.append(None)
                            else:
                                global_control_var.append(grad.clone())
                        average_list(world_size, global_control_var, group)

                    # Sample gradient to decide on clipping.
                    loader_idx = (
                        0 if args.algorithm in ["episode_practical", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"]
                        else 1
                    )
                    if not extra_loaders[loader_idx].has_next():
                        extra_loaders[loader_idx].reset()
                    data = extra_loaders[loader_idx].next_batch()
                    (s1_embed, s2_embed), (s1_lens, s2_lens), targets = data
                    s1_embed, s2_embed = s1_embed.cuda(), s2_embed.cuda()
                    targets = targets.cuda()
                    net.zero_grad()
                    scores = net((s1_embed, s1_lens), (s2_embed, s2_lens))
                    loss = criterion(scores, targets)
                    loss.backward()
                    local_clip_l2_norm = comp_grad_l2_norm(net)

                    # Store sampled local gradients (not just their norms) for
                    # episode_inverted.
                    if args.algorithm in ["episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
                        local_control_var = []
                        for param in net.parameters():
                            if param.grad is None:
                                local_control_var.append(None)
                            else:
                                local_control_var.append(param.grad.clone())

                    # Compute norm of average gradient to decide on clipping.
                    if args.algorithm in ["episode_double", "episode_practical", "episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
                        average_grad(world_size, net, group)
                        global_clip_l2_norm = comp_grad_l2_norm(net)

                    # Store sampled global gradient (not just its norm) for
                    # episode_inverted.
                    if args.algorithm in ["episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
                        global_control_var = []
                        for param in net.parameters():
                            if param.grad is None:
                                global_control_var.append(None)
                            else:
                                global_control_var.append(param.grad.clone())

            # Compute gradient.
            (s1_embed, s2_embed), (s1_lens, s2_lens), targets = data
            s1_embed, s2_embed = s1_embed.cuda(), s2_embed.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            scores = net((s1_embed, s1_lens), (s2_embed, s2_lens))
            loss = criterion(scores, targets)
            loss.backward()

            # Update average gradient across communication round (SCAFFOLD).
            if args.algorithm in ["SCAFFOLD", "episode_practical", "scaffold_clip", "test_correction"]:
                if local_avg_grad is None:
                    local_avg_grad = []
                    for param in net.parameters():
                        if param.grad is not None:
                            local_avg_grad.append(param.grad / args.communication_interval)
                        else:
                            local_avg_grad.append(None)
                else:
                    for i, param in enumerate(net.parameters()):
                        if local_avg_grad[i] is not None:
                            local_avg_grad[i] += param.grad / args.communication_interval

            if 0 == t_total % args.communication_interval:

                if args.test_kappa:
                    with torch.no_grad():
                        for param, param_clone in zip(net.parameters(), net_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, net_clone, group)
                        grad_diff, local_grad, avg_grad = 0, 0, 0
                        for local_param, avg_param in zip(net.parameters(), net_clone.parameters()):
                            if avg_param.grad is not None:
                                avg_grad += float(torch.sum(avg_param.grad * avg_param.grad))
                                local_grad += float(torch.sum(local_param.grad * local_param.grad))
                                diff = avg_param.grad - local_param.grad
                                grad_diff += float(torch.sum(diff * diff))
                        grad_diff = math.sqrt(grad_diff)
                        local_grad = math.sqrt(local_grad)
                        avg_grad = math.sqrt(avg_grad)
                        grad_diffs.append(grad_diff)
                        local_grads.append(local_grad)
                        avg_grads.append(avg_grad)

                if args.algorithm == 'single_clip':
                    average_grad(world_size, net, group)

                elif args.algorithm in ["global_avg_clip", "max_clip"]:
                    with torch.no_grad():
                        for param, param_clone in zip(net.parameters(), net_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, net_clone, group)
                        global_average_grad_l2_norm = comp_grad_l2_norm(net_clone)

            first = (t_total % args.communication_interval == 0)
            _, clip_operation, corrected_operation = optimizer.step(
                global_average_grad_l2_norm,
                prev_local_avg_grad,
                prev_global_avg_grad,
                local_control_var,
                global_control_var,
                local_clip_l2_norm,
                global_clip_l2_norm,
                first=first,
            )
            if clip_operation is not None:
                clip_operations.append(clip_operation)
            if corrected_operation is not None:
                corrected_operations.append(corrected_operation)

            t_total += 1
            t += 1

            # Evaluate the model on training and validation dataset.
            if t in eval_steps:
                eval_idx = eval_steps.index(t)

                elapsed_time = time.time() - eval_start
                eval_elasped_times.append(elapsed_time)

                eval_clip_operations.append(clip_operations)
                eval_corrected_operations.append(corrected_operations)
                clip_operations = []
                corrected_operations = []

                net_clone = copy.deepcopy(net)
                average_model(world_size, net_clone, group)

                # We save and restore the state of the training data loader so that we
                # can iterate over the training set without interrupting the current
                # epoch.
                train_state = train_loader.state()
                train_loss, train_accuracy = evaluate(train_loader, net_clone, criterion)
                all_train_losses.append(train_loss)
                all_train_accuracies.append(train_accuracy)
                train_loader.load_state(train_state)

                test_loss, test_accuracy = evaluate(test_loader, net_clone, criterion)
                all_test_losses.append(test_loss)
                all_test_accuracies.append(test_accuracy)

                print(f'| Rank {args.rank} '
                      f'| GPU {args.gpu_id} '
                      f'| Epoch {epoch_idx} ({eval_idx}) '
                      f'| training time {elapsed_time} seconds '
                      f'| train loss {train_loss:.4f} '
                      f'| train accuracy {train_accuracy:.4f} '
                      f'| test loss {test_loss:.4f} '
                      f'| test accuracy {test_accuracy:.4f} |')

                eval_start = time.time()
                net.train()

        # Decay learning rate.
        if str(epoch_idx) in args.step_decay_milestones:
            print(f'Decay step size and clip param at Epoch {epoch_idx}.')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.step_decay_factor
                if 'clipping_param' in param_group:
                    param_group['clipping_param'] *= args.step_decay_factor

    # Save gradient correction test results to file.
    if args.algorithm == "test_correction":
        attrs = [
            "grad_local_diffs",
            "grad_total_diffs",
            "scaffold_local_diffs",
            "scaffold_total_diffs",
            "episode_local_diffs",
            "episode_total_diffs"
        ]
        for attr in attrs:
            np.save(f"{attr}_{args.rank}", np.array(getattr(optimizer, attr)))

    return_vals = {
        'train_losses': all_train_losses,
        'train_accuracies': all_train_accuracies,
        'test_losses': all_test_losses,
        'test_accuracies': all_test_accuracies,
        'eval_elasped_times': eval_elasped_times,
        'eval_clip_operations': eval_clip_operations,
        'eval_corrected_operations': eval_corrected_operations,
    }
    if args.test_kappa:
        return_vals['grad_diffs'] = grad_diffs
        return_vals['local_grads'] = local_grads
        return_vals['avg_grads'] = avg_grads
    return return_vals, net
