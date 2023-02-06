#!/bin/bash

first_idx=84
family=hetero_silo
eta=0.3
gamma=0.03
heteros=(0.75 0.75 0.85 0.85 0.9 0.9 0.95 0.95 1.0 1.0)
Is=(4 8 4 8 4 8 4 8 4 8)
rounds=(1000 500 1000 500 1000 500 1000 500 1000 500)
algs="local_clip episode_final"

for i in {0..9}; do

    hetero=${heteros[$i]}
    I=${Is[$i]}
    round=${rounds[$i]}

    idx=$(($first_idx + $i))
    name=${idx}_${family}_${I}_${hetero}

    for alg in $algs; do
        bash run.sh $name $alg 1 0 8 $eta $gamma $I $round $hetero
    done
done
