#!/bin/bash
horovodrun -np 8 -H localhost:8 python ../train.py --dataroot /path/to/imagenet --log-dir ../logs/imagenet_all_baselines/fedavg --checkpoint-format checkpoint-fedavg-{epoch}.pth.tar --init-model ../logs/imagenet_all_baselines/init_model.pth --local_steps 128 --batch-size 32 --epochs 90 --base-lr 0.0125 --clipping-param 1e8 --momentum 0.9 --heterogeneity 0.4 --warmup-epochs 0 --optim-method SGDClipGrad
horovodrun -np 8 -H localhost:8 python ../train.py --dataroot /path/to/imagenet --log-dir ../logs/imagenet_all_baselines/local_clip --checkpoint-format checkpoint-local_clip-{epoch}.pth.tar --init-model ../logs/imagenet_all_baselines/init_model.pth --local_steps 128 --batch-size 32 --epochs 90 --base-lr 0.0125 --clipping-param 1.0 --momentum 0.9 --heterogeneity 0.4 --warmup-epochs 0 --optim-method SGDClipGrad
horovodrun -np 8 -H localhost:8 python ../train.py --dataroot /path/to/imagenet --log-dir ../logs/imagenet_all_baselines/scaffold --checkpoint-format checkpoint-scaffold-{epoch}.pth.tar --init-model ../logs/imagenet_all_baselines/init_model.pth --local_steps 128 --batch-size 32 --epochs 90 --base-lr 0.0125 --clipping-param 1e8 --momentum 0.9 --heterogeneity 0.4 --warmup-epochs 0 --optim-method SGDClipGrad --correction scaffold
horovodrun -np 8 -H localhost:8 python ../train.py --dataroot /path/to/imagenet --log-dir ../logs/imagenet_all_baselines/episode --checkpoint-format checkpoint-episode-{epoch}.pth.tar --init-model ../logs/imagenet_all_baselines/init_model.pth --local_steps 128 --batch-size 32 --epochs 90 --base-lr 0.0125 --clipping-param 1.0 --momentum 0.9 --heterogeneity 0.4 --warmup-epochs 0 --optim-method SGDClipGrad --correction episode
