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

    eval_rounds = [
        int((i + 1) * (args.train_rounds / args.num_evals))
        for i in range(args.num_evals)
    ]

    local_control_var = None
    global_control_var = None
    local_clip_l2_norm = None
    global_clip_l2_norm = None

    prev_local_avg_grad = None
    prev_global_avg_grad = None
    local_avg_grad = None
    global_avg_grad = None

    results = {}
    results["train_losses"] = []
    results["train_accuracies"] = []
    results["test_losses"] = []
    results["test_accuracies"] = []
    results["eval_elasped_times"] = []
    results["eval_clip_operations"] = []
    results["eval_corrected_operations"] = []

    clip_operations = []
    corrected_operations = []

    eval_start = time.time()
    for round_idx in range(1, args.train_rounds + 1):

        net.train()

        # Average client models.
        with torch.no_grad():
            average_model(world_size, net, group)

        # Sample clients for each worker.
        current_clients = sample_clients(
            train_loader,
            world_size,
            args.rank,
            train_loader.num_users,
            args.client_sampling,
            args.combined_clients,
            group
        )
        print(current_clients)
        train_loader.set_clients(current_clients)
        test_loader.set_clients(current_clients)
        for extra_loader in extra_loaders:
            extra_loader.set_clients(current_clients)

        # For SCAFFOLD and variants, communicate and reset grad statistics.
        if args.algorithm in ["SCAFFOLD", "episode_practical", "scaffold_clip", "test_correction"] and local_avg_grad is not None:
            scaffold_stats = set_scaffold_stats(
                world_size, local_avg_grad, global_avg_grad, prev_local_avg_grad, prev_global_avg_grad, group
            )
            local_avg_grad = scaffold_stats[0]
            global_avg_grad = scaffold_stats[1]
            prev_local_avg_grad = scaffold_stats[2]
            prev_global_avg_grad = scaffold_stats[3]

        # For EPISODE and variants, compute control variates for each client.
        if args.algorithm in [
            "corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"
        ]:
            episode_stats = set_episode_stats(
                world_size, args.algorithm, extra_loaders, net, criterion, group
            )
            local_control_var = episode_stats[0]
            global_control_var = episode_stats[1]
            local_clip_l2_norm = episode_stats[2]
            global_clip_l2_norm = episode_stats[3]

        # Train for one communication round.
        train_loader, net, optimizer, local_avg_grad, clip_operations, corrected_operations = train_single_round(
            args,
            train_loader,
            net,
            net_clone,
            optimizer,
            criterion,
            local_avg_grad,
            prev_local_avg_grad,
            prev_global_avg_grad,
            local_control_var,
            global_control_var,
            local_clip_l2_norm,
            global_clip_l2_norm,
            clip_operations,
            corrected_operations,
            world_size,
            group
        )

        # Evaluate the model on training and validation dataset.
        if round_idx in eval_rounds:
            eval_idx = eval_rounds.index(round_idx)

            elapsed_time = time.time() - eval_start
            results["eval_elasped_times"].append(elapsed_time)

            results["eval_clip_operations"].append(clip_operations)
            results["eval_corrected_operations"].append(corrected_operations)
            clip_operations = []
            corrected_operations = []

            net_clone = copy.deepcopy(net)
            average_model(world_size, net_clone, group)

            # Evaluate on the entire dataset. To speed up evaluation, comment on the
            # lines below (for both train_loader and test_loader) where we set the
            # clients, though doing so will make it so that evaluation will only happen
            # on participating clients.
            p_size = train_loader.num_users / world_size
            start_client = round(p_size * args.rank)
            end_client = round(p_size * (args.rank + 1))
            eval_clients = list(range(start_client, end_client))
            train_loader.set_clients(eval_clients)
            train_loss, train_accuracy = evaluate(train_loader, net_clone, criterion)
            results["train_losses"].append(train_loss)
            results["train_accuracies"].append(train_accuracy)

            test_loader.set_clients(eval_clients)
            test_loss, test_accuracy = evaluate(test_loader, net_clone, criterion)
            results["test_losses"].append(test_loss)
            results["test_accuracies"].append(test_accuracy)

            total_time = time.time() - eval_start

            msg = (f'| Rank {args.rank} '
                   f'| GPU {args.gpu_id} '
                   f'| Round {round_idx} ({eval_idx}) '
                   f'| training time {elapsed_time:.2f} seconds '
                   f'| total time {total_time:.2f} seconds '
                   f'| train loss {train_loss:.4f} '
                   f'| train accuracy {train_accuracy:.4f} '
                   f'| test loss {test_loss:.4f} '
                   f'| test accuracy {test_accuracy:.4f} |')
            print(msg)

            eval_start = time.time()
            net.train()

        # Decay learning rate.
        if str(round_idx) in args.step_decay_milestones:
            print(f'Decay step size and clip param at Round {round_idx}.')
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

    return results, net


def train_single_round(
    args,
    train_loader,
    net,
    net_clone,
    optimizer,
    criterion,
    local_avg_grad,
    prev_local_avg_grad,
    prev_global_avg_grad,
    local_control_var,
    global_control_var,
    local_clip_l2_norm,
    global_clip_l2_norm,
    clip_operations,
    corrected_operations,
    world_size,
    group,
):

    for t in range(args.communication_interval):
        if not train_loader.has_next():
            train_loader.reset()
        data = train_loader.next_batch()

        # Compute gradient.
        sent_embeds, sent_lens, labels = data
        sent_embeds = sent_embeds.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        scores = net((sent_embeds, sent_lens))
        loss = criterion(scores, labels)
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

        global_average_grad_l2_norm = None
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

        first = (t == 0)
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

    return train_loader, net, optimizer, local_avg_grad, clip_operations, corrected_operations


def sample_clients(train_loader, world_size, rank, num_users, client_sampling, combined_clients, group):
    """ Sample a client (or clients) for each worker. """

    if client_sampling == "random":
        clients = np.random.choice(
            np.arange(num_users),
            size=world_size * combined_clients,
            replace=False
        )
        clients = torch.from_numpy(clients).cuda()
        dist.broadcast(clients, src=0, group=group)
        current_clients = [
            int(clients[combined_clients * rank + i])
            for i in range(combined_clients)
        ]
    elif client_sampling == "silo":
        current_clients = train_loader.silo_clients
    else:
        raise NotImplementedError
    return current_clients


def set_scaffold_stats(world_size, local_avg_grad, global_avg_grad, prev_local_avg_grad, prev_global_avg_grad, group):

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

    return local_avg_grad, global_avg_grad, prev_local_avg_grad, prev_global_avg_grad


def set_episode_stats(world_size, algorithm, extra_loaders, net, criterion, group):

    # Sample gradient to compute control variates for current client.
    if algorithm not in ["episode_practical", "episode_inverted", "episode_final", "delayed_final", "test_correction", "episode_balanced"]:
        if not extra_loaders[0].has_next():
            extra_loaders[0].reset()
        data = extra_loaders[0].next_batch()
        sent_embeds, sent_lens, labels = data
        sent_embeds = sent_embeds.cuda()
        labels = labels.cuda()
        net.zero_grad()
        scores = net((sent_embeds, sent_lens))
        loss = criterion(scores, labels)
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
        0 if algorithm in ["episode_practical", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"]
        else 1
    )
    if not extra_loaders[loader_idx].has_next():
        extra_loaders[loader_idx].reset()
    data = extra_loaders[loader_idx].next_batch()
    sent_embeds, sent_lens, labels = data
    sent_embeds = sent_embeds.cuda()
    labels = labels.cuda()
    net.zero_grad()
    scores = net((sent_embeds, sent_lens))
    loss = criterion(scores, labels)
    loss.backward()
    local_clip_l2_norm = comp_grad_l2_norm(net)

    # Store sampled local gradients (not just their norms) for
    # episode_inverted.
    if algorithm in ["episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
        local_control_var = []
        for param in net.parameters():
            if param.grad is None:
                local_control_var.append(None)
            else:
                local_control_var.append(param.grad.clone())

    # Compute norm of average gradient to decide on clipping.
    if algorithm in ["episode_double", "episode_practical", "episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
        average_grad(world_size, net, group)
        global_clip_l2_norm = comp_grad_l2_norm(net)

    # Store sampled global gradient (not just its norm) for
    # episode_inverted.
    if algorithm in ["episode_inverted", "episode_final", "test_correction", "episode_balanced"]:
        global_control_var = []
        for param in net.parameters():
            if param.grad is None:
                global_control_var.append(None)
            else:
                global_control_var.append(param.grad.clone())

    return local_control_var, global_control_var, local_clip_l2_norm, global_clip_l2_norm
