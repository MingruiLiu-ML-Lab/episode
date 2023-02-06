"""
Train a model on the training set.
"""
import time
import torch
import torch.distributed as dist
from evaluate import evaluate
from sgd_clip import SGDClipGrad

def comp_grad_l2_norm(model) -> float:
    grad_l2_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad_l2_norm_sq += torch.sum(param.grad.data * param.grad.data)
    grad_l2_norm = torch.sqrt(grad_l2_norm_sq).item()
    return grad_l2_norm

def train(args, train_loader, test_loader, net, criterion):
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
                            clipping_param=args.clipping_param, clipping_option='local')

    all_train_losses = []
    all_train_accuracies = []
    all_test_losses = []
    all_test_accuracies = []
    epoch_elasped_times = []
    epoch_clip_operations = []
    for epoch_idx in range(1, args.train_epochs + 1):
        epoch_start = time.time()
        net.train()
        clip_operations = []

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            _, clip_operation = optimizer.step(0.0)
            clip_operations.append(clip_operation)

        # Evaluate the model on training and validation dataset.
        elapsed_time = time.time() - epoch_start
        epoch_elasped_times.append(elapsed_time)

        epoch_clip_operations.append(clip_operations)

        train_loss, train_accuracy = evaluate(train_loader, net, criterion)
        all_train_losses.append(train_loss)
        all_train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = evaluate(test_loader, net, criterion)
        all_test_losses.append(test_loss)
        all_test_accuracies.append(test_accuracy)

        print(f'| Rank {args.rank} '
                f'| GPU {args.gpu_id} '
                f'| Epoch {epoch_idx} '
                f'| training time {elapsed_time} seconds '
                f'| train loss {train_loss:.4f} '
                f'| train accuracy {train_accuracy:.4f} '
                f'| test loss {test_loss:.4f} '
                f'| test accuracy {test_accuracy:.4f} |')

        if str(epoch_idx) in args.step_decay_milestones:
            print(f'Decay step size and clip param at Epoch {epoch_idx}.')
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
