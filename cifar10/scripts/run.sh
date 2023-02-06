#!/bin/bash

if [ "$#" -ne 10 ]; then
    echo "Usage: bash run.sh NAME DATASET ALG TOTAL_NODES NODE WORKERS_PER_NODE I H ETA GAMMA"
    exit
fi

NAME=$1
DATASET=$2
ALG=$3
TOTAL_NODES=$4
NODE=$5
WORKERS_PER_NODE=$6
I=$7
H=$8
ETA=$9
GAMMA=${10}
WORLD_SIZE=$(($WORKERS_PER_NODE * $TOTAL_NODES))
BASE_DIR=../logs/${NAME}
LOG_DIR=${BASE_DIR}/${ALG}

mkdir -p $BASE_DIR
mkdir -p $LOG_DIR

SHAREDFILE="file:///home/op1/projects/federated_clipping/cifar10/logs/${NAME}/${ALG}/sharedfile"
if [ "$ALG" == "fedavg" ]; then
    ALG="local_clip"
    extra=""
    GAMMA=1e8
elif [ "$ALG" == "local_clip" ]; then
    extra=""
elif [ "$ALG" == "fedprox" ]; then
    ALG="local_clip"
    extra="--fedprox --fedprox-mu 0.01"
    GAMMA=1e8
elif [ "$ALG" == "scaffold" ]; then
    extra=""
    GAMMA=1e8
elif [ "$ALG" == "episode" ]; then
    extra=""
else
    echo "Unrecognized algorithm: $ALG."
    exit
fi

if [ "$DATASET" == "CIFAR10" ]; then
    model="resnet32"
    epochs=150
    milestones="80 120"
    interval=5
elif [ "$DATASET" == "MNIST" ]; then
    model="resnet18"
    epochs=20
    milestones="10 15"
    interval=2
else
    echo "Unrecognized dataset: $DATASET."
    exit
fi

i=0
pids=""
while [ $i -lt $WORKERS_PER_NODE ]; do

    rank=$(($NODE * $WORKERS_PER_NODE + $i))
    python ../main.py \
        --init-method $SHAREDFILE \
        --model $model \
        --eta0 $ETA \
        --momentum 0 \
        --weight-decay 5e-4 \
        --step-decay-milestones $milestones \
        --step-decay-factor 0.1 \
        --clipping-param $GAMMA \
        --algorithm $ALG \
        --world-size $WORLD_SIZE \
        --rank $rank \
        --gpu-id $i \
        --communication-interval $I \
        --train-epochs $epochs \
        --eval-interval $interval \
        --batchsize 128 \
        --dataset $DATASET \
        --dataroot ../data \
        --reproducible \
        --seed 0 \
        --heterogeneity $H \
        --log-folder $LOG_DIR \
        --init-model $BASE_DIR/init_model.pth \
        $extra \
        > ${LOG_DIR}/worker_${rank}.out &

    pids="${pids} $!"
    i=$(($i + 1))
done

echo "children:${pids}"
wait
