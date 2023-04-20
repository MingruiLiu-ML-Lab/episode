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

SHAREDFILE="file:///home/mcrawsha/projects/episode/snli/logs/${NAME}/${ALG}/sharedfile"
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

if [ "$DATASET" == "SNLI" ]; then
    epochs=10
    milestones="15 20"
    decay=0.5
    evals=1
    epf_bs_scale=1
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
        --eta0 $ETA \
        --momentum 0 \
        --weight-decay 5e-4 \
        --step-decay-milestones $milestones \
        --step-decay-factor $decay \
        --clipping-param $GAMMA \
        --algorithm $ALG \
        --world-size $WORLD_SIZE \
        --rank $rank \
        --gpu-id $i \
        --communication-interval $I \
        --train-epochs $epochs \
        --evals-per-epoch $evals \
        --batchsize 128 \
        --rnn \
        --n_enc_layers 1 \
        --encoder_dim 2048 \
        --epf_bs_scale $epf_bs_scale \
        --dataset $DATASET \
        --dataroot ../data \
        --reproducible \
        --seed 0 \
        --heterogeneity $H \
        --log-folder $LOG_DIR \
        --init-model $BASE_DIR/init_model.pth \
        > ${LOG_DIR}/worker_${rank}.out &

    pids="${pids} $!"
    i=$(($i + 1))
done

echo "children:${pids}"
wait
