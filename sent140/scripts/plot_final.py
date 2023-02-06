""" Make final figures to appear in paper. """

import os
import argparse
import glob
import json
from math import ceil
from typing import List
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


ALGS = ["local_clip", "episode_final", "global_avg_clip"]

LINE_WIDTH = 1.5
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
I_VALUES = [4]
H_VALUES = [0.8, 0.9, 1.0]
"""
COLORS = {
    **{(1, 0): "black"},
    **{(I, H): color_cycle[i] for i, (I, H) in enumerate(product(I_VALUES, H_VALUES))}
}
"""
COLORS = {
    alg: color_cycle[i] for i, alg in enumerate(ALGS)
}
STYLES = {
    "local_clip": "dashed",
    "episode_final": "solid",
    "global_avg_clip": "dotted",
}
MARKERS = {
    "local_clip": "o",
    "episode_final": "^",
    "global_avg_clip": "s",
}
MARKER_SIZE = 4
TITLES = {
    "train_losses": "Train loss",
    "test_accuracies": "Test accuracy",
    "epochs": "Epoch",
    "rounds": "Round",
    "local_clip": "CELGC",
    "episode_final": "EPISODE",
    "global_avg_clip": "Naive Parallel Clip",
    "steps": "Training Steps",
    "time": "Time",
}
STEPS_PER_EVAL = 800
plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def read_results_dir(results_dir: str):
    avg_paths = glob.glob(os.path.join(results_dir, "*.json"))
    avg_paths = [avg_path for avg_path in avg_paths if "Rank" not in avg_path]
    if not len(avg_paths) == 1:
        print(results_dir)
        print(avg_paths)
    assert len(avg_paths) == 1
    avg_path = avg_paths[0]
    with open(avg_path, "r") as f:
        result = json.load(f)
    return result


def main(log_folders: List[str], output_dir: str):

    # Read results.
    results = {}
    for log_folder in log_folders:
        last_und = log_folder.rfind("_")
        prev_und = log_folder.rfind("_", 0, last_und)
        I = int(log_folder[prev_und+1:last_und])
        H = float(log_folder[last_und+1:])
        for alg in ALGS:
            alg_dir = os.path.join(log_folder, alg)
            if alg != "global_avg_clip":
                key = (alg, I, H)
                if not os.path.isdir(alg_dir):
                    raise ValueError(f"Missing results for setting: {key}")
            else:
                key = (alg, 1, 0)
                if key in results:
                    continue
            results[key] = read_results_dir(alg_dir)
    if ("global_avg_clip", 1, 0) not in results:
        raise ValueError("No logs found for global_avg_clip.")

    # Compute total number of updates in order to compute total number of communication
    # rounds for various I.
    """
    updates_per_epoch = ceil(DATASET_SIZE / TOTAL_BATCH_SIZE)
    num_epochs = max_points // EVALS_PER_EPOCH
    total_updates = updates_per_epoch * num_epochs
    """

    # Plot results.
    keys = list(product(I_VALUES, H_VALUES))
    for axis in ["steps", "time"]:
        for col, (I, H) in enumerate(keys):

            plt.clf()
            fig, ax = plt.subplots(nrows=2, ncols=1)
            fig.set_figheight(10)
            fig.set_figwidth(5)

            for row, metric in enumerate(["train_losses", "test_accuracies"]):
                current_ax = ax[row]
                for alg in ["local_clip", "episode_final"]:
                    key = (alg, I, H)
                    y = results[key][metric]

                    if axis == "steps":
                        x = list(range(len(y)))
                        x = [i * STEPS_PER_EVAL for i in x]
                    elif axis == "time":
                        x = np.cumsum(results[key]["eval_elasped_times"])

                    label = f"{TITLES[alg]}"
                    current_ax.plot(x, y, label=label, color=COLORS[alg], linestyle=STYLES[alg], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
                alg = "global_avg_clip"
                key = (alg, 1, 0)
                y = results[key][metric]
                if axis == "steps":
                    x = list(range(len(y)))
                    x = [i * STEPS_PER_EVAL for i in x]
                elif axis == "time":
                    x = np.cumsum(results[key]["eval_elasped_times"])
                label = f"{TITLES[alg]}"
                current_ax.plot(x, y, label=label, color=COLORS[alg], linestyle=STYLES[alg], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

                # Set axis limits and titles.
                if row == 0 and col == 0:
                    current_ax.legend(fontsize=14)
                current_ax.set_xlabel(TITLES[axis], fontsize=18)
                current_ax.set_ylabel(TITLES[metric], fontsize=18)
                if metric == "train_losses":
                    current_ax.set_ylim([0.47, 0.77])
                elif metric == "test_accuracies":
                    current_ax.set_ylim([0.62, 0.78])
                else:
                    raise NotImplementedError

            fig.savefig(os.path.join(output_dir, f"sent140_{I}_{H}_{axis}.eps"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folders", nargs="+", type=str, help="Folders with logs.")
    parser.add_argument("--output_dir", type=str, default="../logs", help="Output location.")
    args = parser.parse_args()
    main(**vars(args))
