""" Launch distributed training processes. """

import os
import subprocess
import argparse


def main():

    # Get command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="local_clip")
    parser.add_argument("--process_per_node", type=int, default=4)
    parser.add_argument("--total_nodes", type=int, default=2)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--heterogeneity", type=float, default=0.0)
    parser.add_argument("--eta0", type=float, default=0.01)
    parser.add_argument("--clipping-param", type=float, default=1.0)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--n_enc_layers", type=int, default=1)
    parser.add_argument("--unidirectional", default=False, action="store_true")
    parser.add_argument("--rnn", default=False, action="store_true")
    parser.add_argument("--encoder_dim", type=int, default=2048)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--evals-per-epoch", type=int, default=1)
    parser.add_argument("--small", default=False, action="store_true")
    parser.add_argument("--name", type=str, default="experiment")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--test_kappa", default=False, action="store_true")
    parser.add_argument("--epf_bs_scale", type=float, default=1.0)
    parser.add_argument('--step-decay-factor', type=float, default=0.1)
    parser.add_argument('--step-decay-milestones', nargs='*', default=[])
    args = parser.parse_args()

    # Launch training processes.
    world_size = args.process_per_node * args.total_nodes
    pwd = os.getcwd()
    processes = []
    for i in range(args.process_per_node):
        cmd = "python3 ../main.py"
        cmd += " --init-method file://" + pwd + "/sharedfile_" + args.name
        cmd += " --dataroot ../data"
        cmd += " --reproducible"
        cmd += " --seed 0"
        cmd += " --log-folder ../logs"
        cmd += " --validation"

        cmd += " --algorithm " + args.algorithm
        cmd += " --world-size " + str(world_size)
        cmd += " --rank " + str(args.node * args.process_per_node + i)
        cmd += " --gpu-id " + str(i + args.gpu_id)
        cmd += " --communication-interval " + str(args.interval)
        cmd += " --heterogeneity " + str(args.heterogeneity)
        cmd += " --eta0 " + str(args.eta0)
        cmd += " --clipping-param " + str(args.clipping_param)
        cmd += " --batchsize " + str(args.batchsize)
        cmd += " --n_enc_layers " + str(args.n_enc_layers)
        cmd += " --encoder_dim " + str(args.encoder_dim)
        cmd += " --train-epochs " + str(args.train_epochs)
        cmd += " --evals-per-epoch " + str(args.evals_per_epoch)
        cmd += " --epf_bs_scale " + str(args.epf_bs_scale)
        cmd += " --step-decay-factor " + str(args.step_decay_factor)
        if args.unidirectional:
            cmd += " --unidirectional"
        if args.rnn:
            cmd += " --rnn"
        if args.small:
            cmd += " --small"
        if args.test_kappa:
            cmd += " --test_kappa"
        if len(args.step_decay_milestones) > 0:
            cmd += " --step-decay-milestones"
            for step in args.step_decay_milestones:
                cmd += f" {step}"

        processes.append(subprocess.Popen(cmd.split(" ")))

    # Wait for training processes to finish.
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
