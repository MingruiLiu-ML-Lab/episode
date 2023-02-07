# EPISODE: Episodic Gradient Clipping with Periodic Resampled Corrections for Federated Learning with Heterogeneous Data

This repository contains code for the experiments of the paper

[EPISODE: Episodic Gradient Clipping with Periodic Resampled Corrections for Federated Learning with Heterogeneous Data](https://openreview.net/forum?id=ytZIYmztET)
Michael Crawshaw, Yajie Bao, Mingrui Liu, 11th International Conference on Learning Representations, 2023.

### Abstract
Gradient clipping is an important technique for deep neural networks with exploding gradients, such as recurrent neural networks. Recent studies have shown that the loss functions of these networks do not satisfy the conventional smoothness condition, but instead satisfy a relaxed smoothness condition, i.e., the Lipschitz constant of the gradient scales linearly in terms of the gradient norm. Due to this observation, several gradient clipping algorithms have been developed for nonconvex and relaxed-smooth functions. However, the existing algorithms only apply to the single-machine or multiple-machine setting with homogeneous data across machines. It remains unclear how to design provably efficient gradient clipping algorithms in the general Federated Learning (FL) setting with heterogeneous data and limited communication rounds. In this paper, we design EPISODE, the very first algorithm to solve FL problems with heterogeneous data in the nonconvex and relaxed smoothness setting. The key ingredients of the algorithm are two new techniques called \textit{episodic gradient clipping} and \textit{periodic resampled corrections}. At the beginning of each round, EPISODE resamples stochastic gradients from each client and obtains the global averaged gradient, which is used to (1) determine whether to apply gradient clipping for the entire round and (2) construct local gradient corrections for each client. Notably, our algorithm and analysis provide a unified framework for both homogeneous and heterogeneous data under any noise level of the stochastic gradient, and it achieves state-of-the-art complexity results. In particular, we prove that EPISODE can achieve linear speedup in the number of machines, and it requires significantly fewer communication rounds. Experiments on several heterogeneous datasets, including text classification and image classification, show the superior performance of EPISODE over several strong baselines in FL.

### Instructions
We include code to run experiments on CIFAR-10, SNLI, Sent140, ImageNet, and a synthetic objective. Instructions for running experiments on each of these benchmarks can be found in the README of each corresponding subdirectory.

All of the non-synthetic experiments are in Python with PyTorch, and the synthetic experiment uses MATLAB. Most of the Python experiments only require common packages like numpy, though the ImageNet experiments also require [Horovod](https://github.com/horovod/horovod).

### Citation
If you found this repository helpful, please cite our paper:
```
@inproceedings{crawshaw2023episode,
  title={EPISODE: Episodic Gradient Clipping with Periodic Resampled Corrections for Federated Learning with Heterogeneous Data},
  author={Crawshaw, Michael and Bao, Yajie and Liu, Mingrui},
  booktitle={International conference on Learning Representations},
  year={2023}
}
```
