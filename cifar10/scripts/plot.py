""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


LINE_WIDTH = 1.5
TITLES = {
    "train_losses": "Train loss",
    "train_accuracies": "Train Accuracy",
    "test_losses": "Test accuracy",
    "test_accuracies": "Test accuracy",
    "epoch_clip_operations": "Clipping Frequency",
    "epochs": "Epoch",
    "rounds": "Round",
    "local_clip": "CELGC",
    "episode": "EPISODE",
    "fedavg": "FedAvg",
    "scaffold": "SCAFFOLD",
}
EPOCHS_PER_EVAL = 2


plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def plot(log_folder):

    # Read results of each algorithm (averaged over workers).
    results = {}
    metrics = None
    for alg_path, _, alg_files in list(os.walk(log_folder))[1:]:
        alg = os.path.basename(alg_path)
        candidate_avg = [filename for filename in alg_files if "Rank" not in filename and filename.endswith(".json")]
        if len(candidate_avg) != 1:
            print(f"Incomplete results for {alg}, skipping.")
            continue
        avg_filename = candidate_avg[0]
        with open(os.path.join(alg_path, avg_filename), "r") as f:
            results[alg] = json.load(f)

        if metrics is None:
            metrics = list(results[alg].keys())
        else:
            assert metrics == list(results[alg].keys())

    # Re-order methods.
    methods = list(results.keys())
    if set(methods) == set(["scaffold", "local_clip", "episode", "fedavg"]):
        methods = ["fedavg", "local_clip", "scaffold", "episode"]

    # Plot results.
    for metric in ["train_losses", "test_accuracies"]:
        if metric == "epoch_elasped_times":
            continue
        plt.clf()
        for alg in methods:
            x = np.arange(len(results[alg][metric])) * EPOCHS_PER_EVAL
            plt.plot(x, results[alg][metric], label=TITLES[alg], linewidth=LINE_WIDTH)
        plt.xlabel("Epochs")
        plt.ylabel(TITLES[metric])
        plt.legend()

        plt.savefig(os.path.join(log_folder, f"{metric}.eps"), bbox_inches="tight")

    # Print results.
    for alg in results:
        msg = f"{os.path.basename(alg):<12}"
        for metric in metrics:
            if metric != "epoch_elasped_times":
                msg += f" | {metric}: {results[alg][metric][-1]:.5f}"
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_folders", nargs="+", help="Folder containing results to plot", type=str
    )
    args = parser.parse_args()

    for log_folder in args.log_folders:
        print(log_folder)
        plot(log_folder)
