import os
import json
import random
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from arg_parser import arg_parser
from data_loader import Sent140DataLoader
from sent140_net import Sent140Net
from hinge_loss import MultiClassHingeLoss
from train import train


def main():
    args = arg_parser()

    # Make sure that sharedfile does not exist. sharedfile is supposed to be deleted by
    # torch.distributed at the end of training but the package doesn't technically
    # guarantee that this will happen, so we have to make sure ourselves.
    prefix = "file://"
    assert args.init_method.startswith(prefix)
    sharedfile_path = args.init_method[len(prefix):]
    if os.path.isfile(sharedfile_path) and args.rank == 0:
        print(
            "\n========================\n"
            f"Sharedfile {sharedfile_path} already exists. Deleting now.\n"
            "We are assuming that this is an old sharedfile which is fine to delete."
            " If this sharefile is actively being used during another training run,"
            " then you have mistakenly provided the same sharefile path to two"
            " different runs. In this case, deleting the sharedfile (as we are doing"
            " now) will likely destroy the other training run."
            " Be careful to avoid this.\n"
            "========================\n"
        )
        os.remove(sharedfile_path)

    dist.init_process_group(
        backend='nccl',
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(hours=24)
    )

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

    # Load data.
    print('Loading data...')
    if args.algorithm in ["corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double"]:
        extras = [args.communication_interval * args.batchsize, args.batchsize]
    elif args.algorithm in ["episode_practical", "episode_inverted", "delayed_clip", "test_correction", "episode_balanced"]:
        extras = [args.batchsize]
    elif args.algorithm in ["episode_final"]:
        extras = [round(args.batchsize * args.epf_bs_scale)]
    else:
        extras = []

    train_loader = Sent140DataLoader(
        batch_size=args.batchsize,
        root=args.dataroot,
        rank=args.rank,
        world_size=args.world_size,
        split="train",
        small=args.small,
        client_sampling=args.client_sampling,
        silo_hetero=args.silo_hetero,
    )
    test_loader = Sent140DataLoader(
        batch_size=args.batchsize,
        root=args.dataroot,
        rank=args.rank,
        world_size=args.world_size,
        split="test",
        small=args.small,
        client_sampling=args.client_sampling,
        silo_hetero=args.silo_hetero,
    )
    extra_loaders = [
        Sent140DataLoader(
            batch_size=extra_bs,
            root=args.dataroot,
            rank=args.rank,
            world_size=args.world_size,
            split="train",
            small=args.small,
            client_sampling=args.client_sampling,
            silo_hetero=args.silo_hetero,
        )
        for extra_bs in extras
    ]

    # Construct model.
    net = Sent140Net(
        word_embed_dim=train_loader.embed_dim,
        encoder_dim=args.encoder_dim,
        n_enc_layers=args.n_enc_layers,
        dpout_model=args.dpout_model,
        dpout_fc=args.dpout_fc,
        fc_dim=args.fc_dim,
        bsize=args.batchsize,
        n_classes=train_loader.n_classes,
        pool_type=args.pool_type,
        linear_fc=args.linear_fc,
        bidirectional=(not args.unidirectional),
        rnn=args.rnn,
    )

    # Compute model size.
    n_params = sum(p.data.numel() for p in net.parameters())
    print(f"Number of model parameters: {n_params}")

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

    # Train and evaluate the model.
    assert args.algorithm in [
        "fed_avg", "single_clip", "local_clip", "global_avg_clip", "max_clip", "SCAFFOLD", "corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "scaffold_clip", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"
    ]
    print("Training...")
    if args.loss == "svm":
        criterion = MultiClassHingeLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_results, net = train(args, train_loader, test_loader, extra_loaders, net, criterion)

    # Logging results.
    print('Writing the results.')
    if not os.path.exists(args.log_folder) and args.rank == 0:
        os.makedirs(args.log_folder)
    dist.barrier()
    architecture = "RNN" if args.rnn else "LSTM"
    if not args.unidirectional:
        architecture = "Bi" + architecture
    def get_log_name(rank=None):
        log_name = (f'SNLI_SGDClipGrad_'
                   + ('Eta0_%g_' % (args.eta0))
                   + ('Gamma_%g_' % (args.clipping_param))
                   + ('Momentum_%g_' % (args.momentum))
                   + ('Architecture_%s' % (architecture))
                   + ('Layers_%d_' % (args.n_enc_layers))
                   + ('WD_%g_' % (args.weight_decay))
                   + ('Algorithm_%s_' % (args.algorithm))
                   + ('Round_%d_Batchsize_%d_' % (args.train_rounds, args.batchsize))
                   + ('Comm_I_%d' % args.communication_interval))
        if args.algorithm == "episode_final":
            log_name += f"_BSScale_{args.epf_bs_scale}"
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
