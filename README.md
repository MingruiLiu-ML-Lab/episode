## EPISODE

Code for the paper [EPISODE: Episodic Gradient Clipping with Periodic Resampled Corrections for Federated Learning with Heterogeneous Data](https://openreview.net/forum?id=ytZIYmztET) (ICLR 2023).

We include code to run experiments on CIFAR-10, SNLI, Sent140, ImageNet, and a synthetic objective. Instructions for running experiments on each of these benchmarks can be found in the README of each corresponding subdirectory.

All of the non-synthetic experiments are in Python with PyTorch, and the synthetic experiment uses MATLAB. Most of the Python experiments only require common packages like numpy, though the ImageNet experiments also require [Horovod](https://github.com/horovod/horovod).
