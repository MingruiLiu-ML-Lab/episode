#!/bin/bash

name="snli_all_baselines"
algs="local_clip episode_final"
I=4
H=0.7
eta=0.3
gamma=0.03

for alg in $algs; do
    bash run.sh $name SNLI $alg 1 0 8 $I $H $eta $gamma
done
