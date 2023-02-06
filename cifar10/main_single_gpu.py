from arg_parser import arg_parser
from cifar10_resnet import resnet32, resnet56
from data_loader import data_loader
from train_single_gpu import train
import json
import numpy as np
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    args = arg_parser()

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
    dataset = data_loader(dataset_name=args.dataset,
                          dataroot=args.dataroot,
                          batch_size=args.batchsize,
                          val_ratio=(args.val_ratio if args.validation else 0),
                          world_size=1, # Make each GPU load the whole dataset.
                          rank=0)
    train_loader = dataset[0]
    if args.validation:
        test_loader = dataset[1]
    else:
        test_loader = dataset[2]

    if args.model == 'resnet32':
        net = resnet32()
    elif args.model == 'resnet56':
        net = resnet56()
    init_model_path = f"CIFAR10_{args.model}_init_model.pt"
    if os.path.isfile(init_model_path):
        net.load_state_dict(torch.load(init_model_path))
    else:
        torch.save(net.state_dict(), init_model_path)
    net.cuda()

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate the model.
    print("Training...")
    train_results = train(args, train_loader, test_loader, net, criterion)

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    log_name = (f'CIFAR10_{args.model}_SGDClipGrad_'
                + ('Eta0_%g_' % (args.eta0))
                + ('Momentum_%g_' % (args.momentum))
                + ('WD_%g_' % (args.weight_decay))
                + ('Clipping_%s_%g_' % (args.clipping_option, args.clipping_param))
                + ('baseline_' if args.baseline else '')
                + ('Epoch_%d_Batchsize_%d_' % (args.train_epochs, args.batchsize))
                + ('Comm_I_%d_' % args.communication_interval)
                + ('%s' % ('Validation_' if args.validation else 'Test_'))
                + (f'Worldsize_{args.world_size}')
                + (f'Rank_{args.rank}_GPU_{args.gpu_id}'))
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    print('Finished.')


if __name__ == "__main__":
    main()
