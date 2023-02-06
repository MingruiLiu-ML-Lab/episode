#!/bin/bash

name="cifar_all_baselines"
algs="fedavg scaffold local_clip episode"
I=8
H=0.7
eta=0.1
gamma=0.1
init_model=../logs/${name}/init_model.pth

for alg in $algs; do
    bash run.sh $name CIFAR10 $alg 1 0 8 $I $H $eta $gamma &
    while [ ! -f $init_model ]; do
        sleep 1
    done
done
wait
