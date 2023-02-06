#!/bin/bash

first_idx=40
family=full_participation
etas=(0.3 0.3 0.3 0.1 0.1 0.1 0.03 0.03 0.03)
gammas=(0.1 0.3 1.0 0.1 0.3 1.0 0.1 0.3 1.0)
I=5
rounds=1000
algs="local_clip episode_final global_avg_clip"

for i in {0..8}; do

    eta=${etas[$i]}
    gamma=${gammas[$i]}

    idx=$(($first_idx + $i))
    name=${idx}_${family}_${eta}_${gamma}

    for alg in $algs; do
        bash run.sh $name $alg 1 0 16 $eta $gamma $I $rounds
    done
done
