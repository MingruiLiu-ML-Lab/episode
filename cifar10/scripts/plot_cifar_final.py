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
LOC = {
    "train_losses": 1,
    "test_accuracies": 1
}
BBOX_TO_ANCHOR = {
    "train_losses": (1.0, 0.6, 0, 0),
    "test_accuracies": (1.0, 0.8, 0, 0)
}
ZOOM_XLIM = {
    "train_losses": [135, 147.5],
    "test_accuracies": [135, 147.5],
}
ZOOM_YLIM = {
    "train_losses": [0.15, 0.4],
    "test_accuracies": [0.84, 0.91],
}
CORNERS = {
    "train_losses": (3, 4),
    "test_accuracies": (1, 2),
}
EPOCHS_PER_EVAL = 5


plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def main(log_folder):

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

        # Overlay zoomed-in plot.
        zoom = 3
        ax = plt.gca()
        axins = zoomed_inset_axes(
            ax,
            zoom,
            loc=LOC[metric],
            bbox_to_anchor=BBOX_TO_ANCHOR[metric],
            bbox_transform=ax.transAxes,
        )
        for alg in methods:
            x = np.arange(len(results[alg][metric])) * EPOCHS_PER_EVAL
            axins.plot(x, results[alg][metric], label=TITLES[alg], linewidth=LINE_WIDTH)

        """
        # Adjust zoom window.
        ends = [results[method][full_metric][-1] for method in methods]
        y_lim = ZOOM_YLIM[full_metric]
        all_in = all([end > y_lim[0] and end < y_lim[1] for end in ends])
        if not all_in:
            window = max(ends) - min(ends)
            cushion = window * 0.25
            y_lim = [min(ends) - cushion, max(ends) + cushion]
        """

        axins.set_xlim(ZOOM_XLIM[metric])
        axins.set_ylim(ZOOM_YLIM[metric])
        corners = CORNERS[metric]
        mark_inset(ax, axins, loc1=corners[0], loc2=corners[1], fc="none", ec="0.5", linestyle="--")

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
    parser.add_argument("log_folder", help="Folder containing results to plot",
                        type=str)
    args = parser.parse_args()
    main(**vars(args))
