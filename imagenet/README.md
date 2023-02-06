### ImageNet

To run ImageNet experiments, you must first follow the instructions from the [ImageNet Benchmark](https://www.image-net.org/) to download the ImageNet dataset into `imagenet/data`. Then, navigate to `scripts/` and run `bash run_all.sh`. This script takes no arguments and will start running our federated learning experiments with 8 workers and a single set of hyperparameters, for all benchmarks. This script requires that you have installed the [Horovod](https://github.com/horovod/horovod) package. You can plot results with the plotting scripts in `scripts/`.

You can change the hyperparameters (learning rate, clipping parameter, communication interval, data heterogeneity) by editing `run_all.sh`. The heterogeneity argument is inverse to the data similarity parameter `s` that we use in the paper, or specifically `heterogeneity = 1.0 - 100 / s`. Therefore if you want to run with `s = 70%`, set `heterogeneity = 0.3`.

Note that this script will launch 8 worker processes and expect 8 available GPUs indexed 0-7. We do not provide code for training ImageNet over multiple nodes.

Note that it will take several days to complete 90 epochs of training for all four baselines.
