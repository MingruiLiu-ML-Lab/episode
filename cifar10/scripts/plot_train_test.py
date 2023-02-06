""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


def main(log_folder):

    # Read results of each algorithm (averaged over workers).
    results = {}
    for alg, _, alg_files in list(os.walk(log_folder))[1:]:
        candidate_avg = [filename for filename in alg_files if "Rank" not in filename]
        assert len(candidate_avg) == 1
        avg_filename = candidate_avg[0]
        with open(os.path.join(alg, avg_filename), "r") as f:
            results[alg] = json.load(f)

    # Plot results.
    for metric in ["losses", "accuracies"]:
        plt.clf()
        for split in ["train", "test"]:
            full_metric = f"{split}_{metric}"
            for alg in results:
                plt.plot(results[alg][full_metric], label=f"{os.path.basename(alg)}_{split}")
        plt.legend()
        plt.savefig(os.path.join(log_folder, f"{metric}.svg"))

    # Print results.
    for alg in results:
        msg = f"{os.path.basename(alg):<12}"
        for metric in ["train_losses", "test_losses", "train_accuracies", "test_accuracies"]:
            msg += f" | {metric}: {results[alg][metric][-1]:.5f}"
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder", help="Folder containing results to plot",
                        type=str)
    args = parser.parse_args()
    main(**vars(args))
