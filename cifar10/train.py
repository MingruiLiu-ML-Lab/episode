"""
Train a model on the training set.
"""
import copy
import time

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

def average_list(world_size, l, group):
    for param in l:
        dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM, group=group)
        param /= world_size
        dist.broadcast(param.data, src=0, group=group)

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, train_loader, test_loader, extra_loader, net, criterion):
    """
    Args:
        args: parsed command line arguments.
        train_loader: an iterator over the training set.
        test_loader: an iterator over the test set.
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
    prev_global_params = None
    local_correction = None
    global_correction = None
    round_avg_grad = None
    if args.algorithm == "episode":
        extra_iter = cycle(extra_loader)

    epoch_train_losses = []
    epoch_train_accuracies = []
    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    epoch_elasped_times = []
    epoch_clip_operations = []
    t_total = 0
    epoch_start = time.time()
    clip_operations = []
    for epoch in range(1, args.train_epochs + 1):
        net.train()

        for data in train_loader:
            if 0 == t_total % args.communication_interval:
                with torch.no_grad():
                    average_model(world_size, net, group)

                # Store parameters for FedProx.
                if args.fedprox:
                    prev_global_params = [
                        p.clone().detach() for p in net.parameters()
                    ]

                # Update corrections, if necessary.
                if args.algorithm == "scaffold":
                    round_avg_grad, local_correction, global_correction = scaffold_correction(
                        round_avg_grad, world_size, group
                    )
                if args.algorithm == "episode":
                    local_correction, global_correction = episode_correction(
                        net, extra_iter, criterion, world_size, group
                    )

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Compute regularization term for FedProx.
            if args.fedprox:
                params = list(net.parameters())
                assert len(params) == len(prev_global_params)
                for p, gp in zip(params, prev_global_params):
                    loss += args.fedprox_mu * torch.sum((p - gp) ** 2) / 2.0

            loss.backward()

            # Update SCAFFOLD running average.
            if args.algorithm == "scaffold":
                round_avg_grad = update_scaffold_correction(
                    round_avg_grad, net, args.communication_interval
                )

            if 0 == t_total % args.communication_interval:
                if args.algorithm == "global_average_clip":
                    average_grad(world_size, net, group)
                    global_average_grad_l2_norm = comp_grad_l2_norm(net)
                elif args.algorithm in "max_clip":
                    with torch.no_grad():
                        for param, param_clone in zip(net.parameters(), net_clone.parameters()):
                            if param.grad is None:
                                param_clone.grad = None
                            else:
                                param_clone.grad = param.grad.clone()
                        average_grad(world_size, net_clone, group)
                        global_average_grad_l2_norm = comp_grad_l2_norm(net_clone)

            _, clip_operation = optimizer.step(
                global_average_grad_l2_norm,
                local_correction=local_correction,
                global_correction=global_correction,
            )
            clip_operations.append(clip_operation)

            # Track training metrics.
            _, predicted = torch.max(outputs, 1)
            accuracy = (1.0 * (predicted == labels)).mean().item()
            epoch_train_losses.append(loss.item())
            epoch_train_accuracies.append(accuracy)

            t_total += 1

        # Evaluate the model on training and validation dataset.
        if epoch % args.eval_interval == 0:
            elapsed_time = time.time() - epoch_start
            epoch_elasped_times.append(elapsed_time)

            epoch_clip_operations.append(np.mean(clip_operations))
            clip_operations = []

            net_clone = copy.deepcopy(net)
            average_model(world_size, net_clone, group)

            #train_loss, train_accuracy = evaluate(train_loader, net_clone, criterion)
            train_loss = np.mean(epoch_train_losses)
            train_accuracy = np.mean(epoch_train_accuracies)
            all_train_losses.append(train_loss)
            all_train_accuracies.append(train_accuracy)
            epoch_train_losses = []
            epoch_train_accuracies = []

            test_loss, test_accuracy = evaluate(test_loader, net_clone, criterion)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)

            total_time = time.time() - epoch_start

            print(f'| Rank {args.rank} '
                  f'| GPU {args.gpu_id} '
                  f'| Epoch {epoch} '
                  f'| training time {elapsed_time:.2f} seconds '
                  f'| total time {total_time:.2f} seconds '
                  f'| train loss {train_loss:.4f} '
                  f'| train accuracy {train_accuracy:.4f} '
                  f'| test loss {test_loss:.4f} '
                  f'| test accuracy {test_accuracy:.4f} |')

            epoch_start = time.time()

        if str(epoch) in args.step_decay_milestones:
            print(f'Decay step size and clip param at Epoch {epoch}.')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.step_decay_factor
                if 'clipping_param' in param_group:
                    param_group['clipping_param'] *= args.step_decay_factor

    return {'train_losses': all_train_losses,
            'train_accuracies': all_train_accuracies,
            'test_losses': all_test_losses,
            'test_accuracies': all_test_accuracies,
            'epoch_elasped_times': epoch_elasped_times,
            'epoch_clip_operations': epoch_clip_operations}


def scaffold_correction(round_avg_grad, world_size, group):
    """ Compute local and global corrections for SCAFFOLD. """

    if round_avg_grad is None:
        return None, None, None

    local_correction = []
    global_correction = []
    for g in round_avg_grad:
        local_correction.append(g.clone().detach())
        global_correction.append(g.clone().detach())
    average_list(world_size, global_correction, group)

    return None, local_correction, global_correction


def update_scaffold_correction(round_avg_grad, net, communication_interval):
    """ Compute local and global corrections for SCAFFOLD. """

    current_grad = [p.grad for p in net.parameters() if p.grad is not None]
    with torch.no_grad():
        if round_avg_grad is None:
            round_avg_grad = [0] * len(current_grad)
        assert len(current_grad) == len(round_avg_grad)
        for i in range(len(current_grad)):
            round_avg_grad[i] += current_grad[i] / communication_interval

    return round_avg_grad


def episode_correction(net, extra_iter, criterion, world_size, group):
    """ Compute local and global corrections for EPISODE. """

    # Sample batch and compute gradient for local correction.
    inputs, labels = next(extra_iter)
    inputs, labels = inputs.cuda(), labels.cuda()
    net.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    # Average local gradients for global correction.
    local_correction = [
        p.grad.clone().detach() for p in net.parameters() if p.grad is not None
    ]
    global_correction = []
    for g in local_correction:
        global_correction.append(g.clone().detach())
    average_list(world_size, global_correction, group)

    return local_correction, global_correction


def cycle(iterable):
    """ Generator to repeatedly cycle through an iterable. """
    while True:
        for x in iterable:
            yield x
