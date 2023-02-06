import os
import argparse
import math
import time
import random
from tqdm import tqdm
from PIL import ImageFile
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
from horovod.torch.mpi_ops import synchronize, allreduce_async_

from sgd import SGD
from optimizer import SGDClipGrad
from data_loader import data_loader


ImageFile.LOAD_TRUNCATED_IMAGES = True
# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataroot', default=os.path.expanduser('/home/op1/ImageNetData/imagenet'),
                    help='path to training data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--local_steps', type=int, default=1,
                    help='local steps.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--optim-method', type=str, default='SGD',
                    choices=['SGD', 'SGDClipGrad'],
                    help='Which optimizer to use (default: SGD).')
parser.add_argument('--correction', type=str, default=None,
                    choices=['scaffold', 'episode'],
                    help='Which correction to use (default: None).')
parser.add_argument('--fedprox', action='store_true', default=False,
                    help='Add a proximal regularization term as in FedProx.')
parser.add_argument('--fedprox_mu', type=float, default=0.01,
                    help='Regularization coefficient for FedProx.')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--no-sync-warmup', action='store_true', default=False,
                    help='do not synchronize local models during warmup')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--clipping-param', type=float, default=1e10,
                    help='Weight decay used in optimizer (default: 1.0).')
parser.add_argument('--global_grad', action='store_true', default=False,
                    help='Whether to average gradients across workers at each step.')
parser.add_argument('--init-model', type=str, default=None,
                    help='Path to save/load initial model weights.')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--heterogeneity', type=float, default=0.0,
                    help='Data heterogeneity across clients from 0 to 1. (default: 0.0).')
parser.add_argument('--small', action='store_true', default=False,
                    help='use a small version of the dataset')


def allreduce_parameters(params):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the ``model.state_dict()``,
    ``model.named_parameters()``, or ``model.parameters()``.
    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        handle = allreduce_async_(p, name=name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def train(epoch):
    global local_correction, global_correction, steps, prev_global_params

    model.train()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    clip_iterations = 0
    total_iterations = 0

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):

            # Synchronize local models.
            if (
                steps % args.local_steps == 0 or
                (epoch < args.warmup_epochs and not args.no_sync_warmup)
            ):
                allreduce_parameters(model.state_dict())

                # Store parameters at beginning of round for FedProx.
                if args.fedprox:
                    prev_global_params = [
                        p.clone().detach() for p in model.parameters()
                    ]

            # Set local and global correction, if necessary.
            if args.correction is not None and steps % args.local_steps == 0:
                if args.correction == "episode":
                    local_correction, global_correction = episode_correction()
                elif args.correction == "scaffold":
                    local_correction, global_correction = scaffold_correction()
                else:
                    raise NotImplementedError

            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)

                # Average loss among sub-batches.
                loss.div_(math.ceil(float(len(data)) / args.batch_size))

                # Compute regularization term for FedProx.
                if i == 0 and args.fedprox:
                    params = list(model.parameters())
                    assert len(params) == len(prev_global_params)
                    for p, gp in zip(params, prev_global_params):
                        loss += args.fedprox_mu * torch.sum((p - gp) ** 2) / 2.0

                loss.backward()

            # Add gradient to running average for SCAFFOLD.
            if args.correction == "scaffold":
                update_scaffold_correction()

            # Perform optimization step.
            if args.optim_method == 'SGDClipGrad':
                _, clip_operation = optimizer.step(
                    correction=args.correction,
                    local_correction=local_correction,
                    global_correction=global_correction,
                )
                clip_iterations += clip_operation
                total_iterations += 1.0
            else:
                optimizer.step()

            t.set_postfix({'loss': train_loss.avg.item(),'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)
            steps += 1

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        log_writer.add_scalar('train/clipites', clip_iterations, epoch)
        log_writer.add_scalar('train/totalites', total_iterations, epoch)


def validate(epoch):

    # Store worker model, then average worker models for evaluation.
    current_worker = deepcopy(model.state_dict())
    allreduce_parameters(model.state_dict())

    # Evaluate averaged model.
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    # Restore worker model.
    model.load_state_dict(current_worker)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj

        if epoch < args.warmup_epochs:
            param_group['clipping_param'] = 1e10
        else:
            param_group['clipping_param'] = args.clipping_param


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):

    # Store worker model, then average worker models for evaluation.
    current_worker = deepcopy(model.state_dict())
    allreduce_parameters(model.state_dict())

    if hvd.rank() == 0:
        filepath = os.path.join(
            cur_rank_log_dir, args.checkpoint_format.format(epoch=epoch + 1)
        )
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'local_correction': local_correction,
            'global_correction': global_correction,
            'round_avg_grad': round_avg_grad,
        }
        torch.save(state, filepath)

    # Restore worker model.
    model.load_state_dict(current_worker)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def gradient_list():
    """ Return the gradient of the optimizer's parameters as a list. """
    grad_list = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad.data is not None:
                grad_list.append(p.grad.data)
    return grad_list


def episode_correction():
    """ Compute EPISODE corrections. """

    # Sample batch and compute gradient for local correction.
    data, target = next(extra_iter)
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()

    # Store gradient as list, and average local gradients for global correction.
    lc = gradient_list()
    gc = [g.clone() for g in lc]
    allreduce_parameters(gc)

    return lc, gc


def scaffold_correction():
    """ Compute SCAFFOLD corrections. """
    global round_avg_grad

    if round_avg_grad is None:
        return None, None

    # Set local correction to average of gradients from previous round.
    lc = [g.clone() for g in round_avg_grad]
    round_avg_grad = None

    # Average local corrections for global correction.
    gc = [g.clone() for g in lc]
    allreduce_parameters(gc)

    return lc, gc


def update_scaffold_correction():
    """ Update running average of gradients for current round. """
    global round_avg_grad

    current_grad = gradient_list()
    with torch.no_grad():
        if round_avg_grad is None:
            round_avg_grad = [0] * len(current_grad)
        assert len(current_grad) == len(round_avg_grad)
        for i in range(len(current_grad)):
            round_avg_grad[i] += current_grad[i] / args.local_steps


def cycle(iterable):
    """ Generator to repeatedly cycle through an iterable. """
    while True:
        for x in iterable:
            yield x


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Check for valid arguments.
    if args.optim_method != "SGDClipGrad" and args.correction is not None:
        raise ValueError("Can't perform correction unless optim_method='SGDClipGrad'.")

    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # Horovod: write TensorBoard logs on first worker.
    cur_rank_log_dir = os.path.join(args.log_dir, f"rank_{hvd.rank()}")
    if not os.path.exists(cur_rank_log_dir):
        os.makedirs(cur_rank_log_dir)
    log_writer = SummaryWriter(cur_rank_log_dir) # if hvd.rank() == 0 else None

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(
            os.path.join(
                cur_rank_log_dir, args.checkpoint_format.format(epoch=try_epoch)
            )
        ):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(4)

    # Load dataset.
    validation = False
    val_ratio = 0
    extra_bs = None
    if args.correction == "episode":
        extra_bs = round(allreduce_batch_size * 8)
    dataset = data_loader(
        dataroot=args.dataroot,
        batch_size=allreduce_batch_size,
        val_ratio=val_ratio,
        world_size=hvd.size(),
        rank=hvd.rank(),
        heterogeneity=args.heterogeneity,
        num_workers=4,
        small=args.small,
        extra_bs=extra_bs,
    )
    train_loader = dataset[0]
    val_loader = dataset[1] if validation else dataset[2]
    extra_loader = dataset[3]
    if extra_loader is not None:
        extra_iter = cycle(extra_loader)

    # Set up standard ResNet-50 model.
    model = models.resnet50()
    if args.init_model is not None:

        # Save model if necessary.
        if not os.path.isfile(args.init_model) and hvd.rank() == 0:
            if not os.path.isdir(os.path.dirname(args.init_model)):
                os.makedirs(os.path.dirname(args.init_model))
            torch.save(model.state_dict(), args.init_model)

        # Dummy sync to make sure all workers wait for first to store model.
        hvd.allreduce(torch.tensor(0), name="Barrier")

        # Load model.
        model.load_state_dict(torch.load(args.init_model))

    # By default, Adasum doesn't need scaling up learning rate.
    # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = args.batches_per_allreduce * hvd.local_size()

    # Horovod: scale learning rate by the number of GPUs.
    if args.optim_method == 'SGD':
        optimizer = SGD(model.parameters(),
                        lr=(args.base_lr * lr_scaler),
                        momentum=args.momentum,
                        weight_decay=args.wd,
                        clipping_param=args.clipping_param)
    elif args.optim_method == 'SGDClipGrad':
        optimizer = SGDClipGrad(model.parameters(),
                                lr=(args.base_lr * lr_scaler),
                                momentum=args.momentum,
                                weight_decay=args.wd,
                                clipping_param=args.clipping_param)

    # Initialize local and global corrections for SCAFFOLD-style updates.
    local_correction = None
    global_correction = None
    round_avg_grad = None

    # Initialize previous round parameters for FedProx.
    prev_global_params = None

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    if args.global_grad:
        assert args.local_steps == 1
        assert args.optim_method == 'SGD'
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=args.batches_per_allreduce,
            op=hvd.Adasum if args.use_adasum else hvd.Average,
            gradient_predivide_factor=args.gradient_predivide_factor
        )

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = os.path.join(
            cur_rank_log_dir, args.checkpoint_format.format(epoch=resume_from_epoch)
        )
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        local_correction = checkpoint['local_correction']
        global_correction = checkpoint['global_correction']
        round_avg_grad = checkpoint['round_avg_grad']

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    steps = 0
    for epoch in range(resume_from_epoch, args.epochs):
        epoch_start_time = time.time()
        train(epoch)
        epoch_elapsed_time = time.time() - epoch_start_time
        log_writer.add_scalar('train/time', epoch_elapsed_time, epoch)
        validate(epoch)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch)
