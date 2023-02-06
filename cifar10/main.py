from arg_parser import arg_parser
from cifar10_resnet import resnet32, resnet56, mnist_resnet18
from data_loader import data_loader
from train import train
import json
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    args = arg_parser()
    dist.init_process_group(backend='nccl',
                            init_method=args.init_method,
                            world_size=args.world_size,
                            rank=args.rank)

    torch.cuda.set_device(args.gpu_id)
    print(f"| Rank {args.rank} | Requested GPU {args.gpu_id} "
          f'| Assigned GPU {torch.cuda.current_device()} |')

    # Set the ramdom seed for reproducibility.
    if args.reproducible:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    # Load data, note we will also call the validation set as the test set.
    print('Loading data...')
    extra_bs = None
    if args.algorithm == "episode":
        extra_bs = args.batchsize * args.communication_interval
    dataset = data_loader(dataset_name=args.dataset,
                          dataroot=args.dataroot,
                          batch_size=args.batchsize,
                          val_ratio=(args.val_ratio if args.validation else 0),
                          world_size=args.world_size,
                          rank=args.rank,
                          heterogeneity=args.heterogeneity,
                          extra_bs=extra_bs,
                          small=args.small)
    train_loader = dataset[0]
    if args.validation:
        test_loader = dataset[1]
    else:
        test_loader = dataset[2]
    extra_loader = dataset[3]

    if args.model == 'resnet18':
        assert args.dataset == "MNIST"
        net = mnist_resnet18()
    elif args.model == 'resnet32':
        assert args.dataset.startswith("CIFAR")
        net = resnet32()
    elif args.model == 'resnet56':
        assert args.dataset.startswith("CIFAR")
        net = resnet56()
    elif args.model == 'logreg':
        assert args.dataset == "MNIST"
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 10)
        )

    # Initialize or load model weights.
    if not os.path.isfile(args.init_model) and args.rank == 0:
        print("Initializing model weights from scratch.")
        if not os.path.isdir(os.path.dirname(args.init_model)):
            os.makedirs(os.path.dirname(args.init_model))
        torch.save(net.state_dict(), args.init_model)
    dist.barrier()
    print("Loading initial model weights.")
    net.load_state_dict(torch.load(args.init_model))
    net.cuda()

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate the model.
    print("Training...")
    train_results = train(args, train_loader, test_loader, extra_loader, net, criterion)

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder) and args.rank == 0:
        os.makedirs(args.log_folder)
    dist.barrier()
    def get_log_name(rank=None):
        log_name = (f'CIFAR10_{args.model}_SGDClipGrad_'
                + ('Eta0_%g_' % (args.eta0))
                + ('Momentum_%g_' % (args.momentum))
                + ('WD_%g_' % (args.weight_decay))
                + ('Algorithm_%s_' % (args.algorithm))
                + ('Gamma_%g_' % (args.clipping_param))
                + ('Epoch_%d_Batchsize_%d_' % (args.train_epochs, args.batchsize))
                + ('Comm_I_%d_' % args.communication_interval)
                + ('%s' % ('Validation' if args.validation else 'Test')))
        if rank is not None:
            log_name += f'_Rank_{rank}'
        return log_name
    log_name = get_log_name(args.rank)
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    # Log average results.
    dist.barrier()
    if args.rank == 0:
        client_results = []
        for rank in range(args.world_size):
            log_name = get_log_name(rank)
            with open(f"{args.log_folder}/{log_name}.json", "r") as f:
                client_results.append(json.load(f))
        keys = list(client_results[0].keys())
        for client_result in client_results[1:]:
            assert keys == list(client_result.keys())

        avg_results = {}
        for key in keys:
            avg_results[key] = np.mean(
                [client_result[key] for client_result in client_results],
                axis=0
            ).tolist()
        log_name = get_log_name()
        with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
            json.dump(avg_results, f)

    print('Finished.')


if __name__ == "__main__":
    main()
