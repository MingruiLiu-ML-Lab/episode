#!/bin/bash

if [ "$#" -ne 10 ]; then
    echo "Usage: bash run.sh NAME ALG TOTAL_NODES NODE WORKERS_PER_NODE ETA GAMMA I ROUNDS SILO_HETERO"
    exit
fi

NAME=$1
ALG=$2
TOTAL_NODES=$3
NODE=$4
WORKERS_PER_NODE=$5
ETA=$6
GAMMA=$7
I=$8
ROUNDS=$9
SILO_HETERO=${10}
WORLD_SIZE=$(($WORKERS_PER_NODE * $TOTAL_NODES))
BASE_DIR=../logs/${NAME}
LOG_DIR=${BASE_DIR}/${ALG}
NUM_GPUS=8

if [ "$ALG" == "global_avg_clip" ]; then
    ROUNDS=$(($ROUNDS * $I))
    I=1
fi

first_m=$(($ROUNDS / 2))
second_m=$((3 * $ROUNDS / 4))
milestones="$first_m $second_m"
decay=0.5
evals=10
client_sampling="silo"
combined_clients=1
epf_bs_scale=1

mkdir -p $BASE_DIR
mkdir -p $LOG_DIR

SHAREDFILE="file:///home/op1/projects/federated_clipping/sent140/logs/${NAME}/${ALG}/sharedfile"
if [ "$ALG" == "fedavg" ]; then
    ALG="local_clip"
    GAMMA=1e8
elif [ "$ALG" == "local_clip" ]; then
    :
elif [ "$ALG" == "SCAFFOLD" ]; then
    GAMMA=1e8
elif [ "$ALG" == "episode_final" ]; then
    :
elif [ "$ALG" == "global_avg_clip" ]; then
    if [ "$I" != "1" ]; then
        echo "I must be set to 1 if using global_avg_clip."
        echo "Given value of I: ${I}"
        exit
    fi
else
    echo "Unrecognized algorithm: $ALG."
    exit
fi

i=0
pids=""
while [ $i -lt $WORKERS_PER_NODE ]; do

    gpu=$i
    while [ $gpu -ge $NUM_GPUS ]; do
        gpu=$(($gpu - $NUM_GPUS))
    done

    rank=$(($NODE * $WORKERS_PER_NODE + $i))
    python ../main.py \
        --init-method $SHAREDFILE \
        --eta0 $ETA \
        --momentum 0 \
        --weight-decay 5e-4 \
        --step-decay-milestones $milestones \
        --step-decay-factor $decay \
        --clipping-param $GAMMA \
        --algorithm $ALG \
        --world-size $WORLD_SIZE \
        --rank $rank \
        --gpu-id $gpu \
        --communication-interval $I \
        --train-rounds $ROUNDS \
        --num-evals $evals \
        --batchsize 128 \
        --n_enc_layers 2 \
        --encoder_dim 256 \
        --epf_bs_scale $epf_bs_scale \
        --dataroot ../data \
        --client_sampling $client_sampling \
        --combined_clients $combined_clients \
        --silo-hetero $SILO_HETERO \
        --reproducible \
        --seed 0 \
        --log-folder $LOG_DIR \
        --init-model $BASE_DIR/init_model.pth \
        > ${LOG_DIR}/worker_${rank}.out &

    pids="${pids} $!"
    i=$(($i + 1))
done

echo "children:${pids}"
wait
