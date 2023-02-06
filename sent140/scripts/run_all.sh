#!/bin/bash

name="sent140_all_baselines"
algs="local_clip episode_final global_avg_clip"
I=4
H=0.9
eta=0.3
gamma=0.03
rounds=1000

for alg in $algs; do
    bash run.sh $name $alg 1 0 8 $eta $gamma $I $rounds $H
done
