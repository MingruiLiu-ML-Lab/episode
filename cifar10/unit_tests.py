import json
import os
import torch
from arg_parser import arg_parser
from data_loader import data_loader
from cifar10_resnet import resnet56
from sgd_clip import SGDClipGrad
import torch.nn as nn

def test_arg_parser():
    args = arg_parser()
    print(args)

def test_data_loader():
    dataset_name = 'CIFAR10'
    dataroot = 'data'
    batch_size = 50
    val_ratio = 0.2
    world_size = 8

    tr0, val0, ts0 = data_loader(dataset_name, dataroot, batch_size, val_ratio, world_size, rank=0)
    assert(len(tr0) == 100)
    assert(len(val0) == 13)
    assert(len(ts0) == 100)
    tr4, val4, ts4 = data_loader(dataset_name, dataroot, batch_size, val_ratio=0, world_size=8, rank=4)
    assert(len(tr4) == 125)
    assert(len(val4) == 0)
    assert(len(ts4) == 100)

    tr, val, ts = data_loader(dataset_name, dataroot, batch_size, val_ratio=0, world_size=1, rank=0)
    assert(len(tr) == 1000)
    assert(len(val) == 0)
    assert(len(ts) == 100)

    print('Finished')


def test_sgd_clip():
    net = resnet56()
    optimizer = SGDClipGrad(net.parameters(), lr=0.1, clipping_param=1.0, clipping_option='global_average')
    train_loader, val_loader, test_loader = data_loader('CIFAR10', dataroot='data',
                                                        batch_size=128, val_ratio=0,
                                                        world_size=8, rank=1)
    for input, label in train_loader:
        optimizer.zero_grad()
        outputs = net(input)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step(global_average_grad_l2_norm=100.0)
    
    print('Finished')

def test_print_info():
    args = arg_parser()
    epoch = 10
    elapsed_time = 165
    train_loss = 1.341234
    train_accuracy = 0.893754
    test_loss = 2.13445
    test_accuracy = 0.973
    print(f'| Rank {args.rank} | GPU {args.gpu_id}| Epoch {epoch} '
            f'| training time {elapsed_time} seconds '
            f'| train loss {train_loss:.4f} '
            f'| train accuracy {train_accuracy:.4f} '
            f'| test loss {test_loss:.4f} '
            f'| test accuracy {test_accuracy:.4f} |')

def test_gpu_info():
    args = arg_parser()
    torch.cuda.set_device(args.gpu_id)
    print(f"| Rank {args.rank} | Requested GPU {args.gpu_id} "
          f'| Assigned GPU {torch.cuda.current_device()} ')

def test_save_log():
    train_results = {'train_losses': [1, 2.3],
                    'train_accuracies': [0.9, 0.7],
                    'test_losses': [3.1, 190],
                    'test_accuracies': [0.3, 0.05],
                    'epoch_elasped_times': [14, 16],
                    'epoch_clip_operations': [[0, 0, 1], [1, 0, 0]]}
    args = arg_parser()    
    print('Writing the results.')
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    log_name = ('CIFAR10_resnet56_SGDClipGrad_'
                + ('Eta0_%g_' % (args.eta0))
                + ('Momentum_%g_' % (args.momentum))
                + ('WD_%g_' % (args.weight_decay))
                + ('Clipping_%s_%g_' % (args.clipping_option, args.clipping_param))
                + ('Epoch_%d_Batchsize_%d_' % (args.train_epochs, args.batchsize))
                + ('%s' % ('Validation_' if args.validation else 'Test_'))
                + (f'Rank_{args.rank}_GPU_{args.gpu_id}'))
    with open(f"{args.log_folder}/{log_name}.json", 'w') as f:
        json.dump(train_results, f)

    with open(f"{args.log_folder}/{log_name}.json", 'r') as f:
        read_stats = json.load(f)
        assert(read_stats == train_results)
        print(read_stats)

if __name__ == '__main__':
    # test_arg_parser()
    # test_data_loader()
    # test_sgd_clip()
    # test_print_info()
    # test_gpu_info()
    # test_save_log()
    print('Finished')
