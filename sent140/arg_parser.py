"""
Command-line argument parsing.
"""

import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Distributed Training SGDClipGrad on CIFAR10 with Resnet26')
    parser.add_argument('--eta0', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1).')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use nesterov momentum (default: False).')
    parser.add_argument('--momentum', type=float, default=0,
                        help='Momentum used in optimizer (default: 0).')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='Weight decay used in optimizer (default: 0).')
    parser.add_argument('--step-decay-milestones', nargs='*', default=[],
                        help='Used for step decay denoting when to decrease the step size and the clipping parameter, unit in iteration (default: []).')   
    parser.add_argument('--step-decay-factor', type=float, default=0.1,
                        help='Step size and clipping paramter shall be multiplied by this on step decay (default: 0.1).')
    parser.add_argument('--clipping-param', type=float, default=5.0,
                        help='Weight decay used in optimizer (default: 5.0).')
    parser.add_argument('--algorithm', type=str, default='single_clip',
                        help='Optimization algorithm for training. Choices: ["single_clip", "local_clip", "max_clip", "global_avg_clip", "SCAFFOLD", "corrected_clip", "episode_scaffold", "episode_normal_1", "episode_double", "episode_practical", "scaffold_clip", "episode_inverted", "episode_final", "delayed_clip", "test_correction", "episode_balanced"].')

    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of processes in training (default: 1).')
    parser.add_argument('--rank', type=int, default=0,
                        help='Which process is this (default: 0).')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU is used in this process (default: 0).')
    parser.add_argument('--init-method', type=str, default='file://',
                        help='URL specifying how to initialize the process group (default: file//).')  
    parser.add_argument('--communication-interval', type=int, default=8,
                        help='Number of training steps between communication (default: 8).')

    # Training
    parser.add_argument('--train-rounds', type=int, default=50,
                        help='Number of communication rounds (default: 50).')
    parser.add_argument('--batchsize', type=int, default=32,
                        help='How many images in each train step for each GPU (default: 32).')
    parser.add_argument("--num-evals", type=int, default=1,
                        help="How many times to evaluate model during training.")
    parser.add_argument('--init-model', type=str, default='../logs/init_model.pth',
                        help='Path to store/load the initial weights (default: ../logs/init_model.pth).')

    parser.add_argument('--dataroot', type=str, default='../data',
                        help='Where to retrieve data (default: ../data).')
    parser.add_argument("--small", default=False, action='store_true', help="Use smaller version of dataset.")
    parser.add_argument("--client_sampling", type=str, default="random", help="Method of client sampling. Choices: ['random', 'silo'].")
    parser.add_argument("--combined_clients", type=int, default=1, help="Number of clients to combine onto each device. Only used when client_sampling='random'.")
    parser.add_argument("--silo-hetero", type=float, default=0.0, help="Heterogeneity of client partitioning into silos. Only used when client_sampling='silo'.")

    parser.add_argument("--encoder_dim", type=int, default=2048, help="encoder nhid dimension")
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
    parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")
    parser.add_argument("--linear_fc", default=False, action='store_true', help="don't use nonlinearity in fc")
    parser.add_argument("--unidirectional", default=False, action='store_true', help="don't use bidirectional recurrent network.")
    parser.add_argument("--rnn", default=False, action='store_true', help="don't use LSTM, use vanilla RNN instead.")

    parser.add_argument("--loss", type=str, default='svm', help="choice of loss function")
    parser.add_argument('--reproducible', action='store_true',
                        help='Ensure reproducibility (default: False).')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='Random seed (default: 0).')

    parser.add_argument('--log-folder', type=str, default='../logs',
                        help='Where to store results.')

    parser.add_argument('--epf_bs_scale', type=float, default=1.0, help='Multiplier for batch size in episode_final when computing G_r and G_r^i.')

    return parser.parse_args()
