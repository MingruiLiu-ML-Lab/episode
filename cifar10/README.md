### CIFAR-10

To run CIFAR-10 experiments, navigate to `scripts/` and run `bash run_all.sh`. This script takes no arguments and will start running our federated learning experiments with 8 workers and a single set of hyperparameters. You can plot results with the plotting scripts in `scripts/`.

You can change the hyperparameters (learning rate, clipping parameter, communication interval, data heterogeneity) by editing `run_all.sh`. The data heterogeneity variable `H` is inverse to the data similarity parameter `s` that we use in the paper, or specifically `H = 1.0 - 100 / s`. Therefore if you want to run with `s = 70%`, set `H = 0.3`.

Note that this script will launch 8 worker processes and expect 8 available GPUs indexed 0-7, though you can edit `run_all.sh` to run this script over multiple nodes (e.g. 2 nodes with 4 GPUs each). To do this, create 2 copies of the script (one for each node) and edit the `1 0 8` arguments to `run.sh` on line 12 to `2 0 4` in the first copy and `2 1 4` in the second copy. Then run each script on it's corresponding node simultaneously. Other multi-node patterns can be configured accordingly.

It can be annoying to start and stop training many times, since you'll have to manually kill the child processes each time you want to stop training. Instead of CTRL-C'ing to stop execution of `run_all.sh`, it's easier to just run `bash kill_last_job_and_children.sh`.
