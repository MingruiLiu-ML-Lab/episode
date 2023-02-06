""" Make final figures to appear in paper. """

import os
import argparse
import glob
import json
from math import ceil
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


LOG_DIR = "./"
LOG_TEMPLATES = ["SNLI_SGDClipGrad_Eta0_0.*eity_0.7*.json", "SNLI_SGDClipGrad_Eta0_0.*global_avg_clip*.json"]

ALGS = ["local_clip", "episode_final", "global_avg_clip"]

LINE_WIDTH = 1.5
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
I_VALUES = [1, 2, 4, 8, 16]
COLORS = {
    **{1: "black"},
    **{I_VALUES[i]: color_cycle[i] for i in range(1, len(I_VALUES))},
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
    "test_accs": "Test accuracy",
    "epochs": "Epoch",
    "rounds": "Round",
    "local_clip": "CELGC",
    "episode_final": "EPISODE",
    "global_avg_clip": "Naive Parallel Clip",
}
LOC = {
    "train_losses": 1,
    "test_accs": 4,
}
BBOX_TO_ANCHOR = {
    "train_losses": (1.0, 1.0, 0, 0),
    "test_accs": (1.0, 0.1, 0, 0),
}
ZOOM_XLIM = {
    "train_losses": [500, 5000],
    "test_accs": [500, 5000]
}
ZOOM_YLIM = {
    "train_losses": [0.325, 0.475],
    "test_accs": [0.7925, 0.8225]
}
CORNERS = {
    "train_losses": (2, 4),
    "test_accs": (1, 3),
}
plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})

DATASET_SIZE = 549367
TOTAL_BATCH_SIZE = 512
EVALS_PER_EPOCH = 1

TEMP_TYPE = "zoom"


def read_results_dir(results_dir: str):
    return None


def main(log_folders: List[str], output_dir: str):

    # Read results.
    results = {}
    for log_folder in log_folders:
        last_und = log_folder.rfind("_")
        prev_und = log_folder.rfind("_", 0, last_und)
        I = int(log_folder[prev_und+1:last_und])
        H = float(log_folder[last_und+1:])
        print(I, H)
        for alg in ALGS:
            alg_dir = os.path.join(log_folder, alg)
            if alg != "global_avg_clip":
                key = (alg, I, H)
                if not os.path.isdir(alg_dir):
                    print(alg_dir)
                    raise ValueError(f"Missing results for setting: {key}")
            else:
                key = (alg, 1, 0)
                if os.path.isdir(alg_dir) and key in results:
                    raise ValueError("Duplicate logs found for global_avg_clip.")
            results[key] = read_results_dir(alg_dir)
    if ("global_avg_clip", 1, 0) not in results:
        raise ValueError("No logs found for global_avg_clip.")

    print("made it!")
    exit()

    # Read in training loss and testing accuracy.
    results = []
    for log_folder in log_folders:
        for alg in ALGS:
            pass

    # Get names of results from each tuning run.
    total_names = []
    for template in LOG_TEMPLATES:
        total_names += glob.glob(os.path.join(LOG_DIR, template))
    unique_names = []
    for name in total_names:
        rank_pos = name.find("Rank")
        aggregate_name = name[:rank_pos-1]
        if aggregate_name not in unique_names:
            unique_names.append(aggregate_name)

    # Average results over all workers.
    results = {
        "train_losses": {},
        "test_losses": {},
        "train_accs": {},
        "test_accs": {},
        "clipping": {},
        "elasped_times": {}
    }
    for aggregate_name in unique_names:

        # Construct key to name run.
        eta_pos = aggregate_name.find("Eta0")
        under_pos_1 = aggregate_name.find("_", eta_pos)
        under_pos_2 = aggregate_name.find("_", under_pos_1 + 1)
        eta = float(aggregate_name[under_pos_1+1:under_pos_2])
        gamma_pos = aggregate_name.find("Gamma")
        if gamma_pos == -1:
            gamma = None
        else:
            under_pos_1 = aggregate_name.find("_", gamma_pos)
            under_pos_2 = aggregate_name.find("_", under_pos_1 + 1)
            gamma = float(aggregate_name[under_pos_1+1:under_pos_2])
        alg_pos = aggregate_name.find("Algorithm")
        under_pos = aggregate_name.find("_", alg_pos)
        epoch_pos = aggregate_name.find("Epoch")
        alg = aggregate_name[under_pos+1:epoch_pos-1]
        key = f"{alg}_{eta}_{gamma}"
        under_pos_1 = aggregate_name.find("_I_") + 2
        under_pos_2 = aggregate_name.find("_", under_pos_1 + 1)
        I = aggregate_name[under_pos_1+1: under_pos_2]
        key += f"_{I}"

        # Read results for each worker and average them.
        worker_names = glob.glob(os.path.join(LOG_DIR, aggregate_name + "*"))
        worker_train_losses = []
        worker_test_losses = []
        worker_train_accs = []
        worker_test_accs = []
        worker_clipping = []
        worker_times = []
        for worker_name in worker_names:
            with open(worker_name, "r") as worker_file:
                worker_results = json.load(worker_file)
            worker_train_losses.append(worker_results["train_losses"])
            worker_test_losses.append(worker_results["test_losses"])
            worker_train_accs.append(worker_results["train_accuracies"])
            worker_test_accs.append(worker_results["test_accuracies"])
            #worker_clipping.append(np.mean(worker_results["eval_clip_operations"]))
            worker_clipping.append([])
            for eval_clip in worker_results["eval_clip_operations"]:
                worker_clipping[-1].append(np.mean(eval_clip))
            worker_times.append(worker_results["eval_elasped_times"])

        results["train_losses"][key] = np.mean(np.array(worker_train_losses), axis=0)
        results["test_losses"][key] = np.mean(np.array(worker_test_losses), axis=0)
        results["train_accs"][key] = np.mean(np.array(worker_train_accs), axis=0)
        results["test_accs"][key] = np.mean(np.array(worker_test_accs), axis=0)
        results["clipping"][key] = np.mean(np.array(worker_clipping), axis=0)
        results["elasped_times"][key] = np.mean(np.array(worker_times), axis=0)

        # Smooth out clipping results by taking local averages.
        avg_window = 10
        smooth_results = np.zeros_like(results["clipping"][key])
        for i in range(len(results["clipping"][key])):
            lower = max(0, i - avg_window)
            upper = min(len(results["clipping"][key] - 1), i + avg_window)
            smooth_results[i] = np.mean(results["clipping"][key][lower:upper+1])
        results["clipping"][key] = smooth_results.copy()

    # Collect maximum number of points to plot across runs.
    keys = list(results["train_losses"].keys())
    max_points = {"train_losses": 0, "test_losses": 0, "train_accs": 0, "test_accs": 0, "clipping": 0}
    for metric in ["train_losses", "test_losses", "train_accs", "test_accs", "clipping"]:
        for key in keys:
            max_points[metric] = max(max_points[metric], len(results[metric][key]))
    assert len(set(max_points.values())) == 1
    max_points = max_points["train_losses"]

    # Compute total number of updates in order to compute total number of communication
    # rounds for various I.
    updates_per_epoch = ceil(DATASET_SIZE / TOTAL_BATCH_SIZE)
    num_epochs = max_points // EVALS_PER_EPOCH
    total_updates = updates_per_epoch * num_epochs

    # Plot results.
    for i, axis in enumerate(["epochs", "rounds"]):
        plt.clf()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        fig.set_figheight(10)
        fig.set_figwidth(5)
        axins = []
        for j, metric in enumerate(["train_losses", "test_accs"]):
            sub_idx = 2 * j + i
            current_ax = ax[j]
            for alg in ["local_clip", "episode_final"]:
                for I in [2, 4, 8, 16]:
                    candidate_keys = [key for key in keys if key.startswith(alg) and key.endswith(str(I))]
                    assert len(candidate_keys) == 1
                    key = candidate_keys[0]
                    if axis == "epochs":
                        x = [(i + 1) * max_points / len(results[metric][key]) for i in range(len(results[metric][key]))]
                    elif axis == "rounds":
                        total_rounds = total_updates // I
                        x = [(i+1) * total_rounds / len(results[metric][key]) for i in range(len(results[metric][key]))]
                    else:
                        raise NotImplementedError
                    label = f"{TITLES[alg]}, I = {I}"
                    current_ax.plot(x, results[metric][key], label=label, color=COLORS[I], linestyle=STYLES[alg], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)
            alg = "global_avg_clip"
            I = 1
            single_keys = [key for key in keys if key.startswith("global_avg_clip")]
            assert len(single_keys) == 1
            single_key = single_keys[0]
            if axis == "epochs":
                x = [(i + 1) * max_points / len(results[metric][single_key]) for i in range(len(results[metric][single_key]))]
            elif axis == "rounds":
                total_rounds = total_updates // I
                x = [(i+1) * total_rounds / len(results[metric][single_key]) for i in range(len(results[metric][single_key]))]
            else:
                raise NotImplementedError
            label = f"{TITLES[alg]}"
            current_ax.plot(x, results[metric][single_key], label=label, color=COLORS[I], linestyle=STYLES[alg], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

            # Set axis limits and titles.
            if sub_idx == 0:
                current_ax.legend(fontsize=9)
            current_ax.set_xlabel(TITLES[axis], fontsize=18)
            current_ax.set_ylabel(TITLES[metric], fontsize=18)
            if metric == "train_losses":
                current_ax.set_ylim([0.15, 0.7])
            elif metric == "test_accs":
                current_ax.set_ylim([0.74, 0.84])
            else:
                raise NotImplementedError

            # Log scale.
            if TEMP_TYPE == "log":
                if axis == "rounds":
                    current_ax.set_xscale("log")
                    current_ax.set_xlim([781, 25000])
                    current_ax.set_xticks([1562, 3125, 6250, 12500])
                    current_ax.set_xticklabels([1562, 3125, 6250, 12500])
                    current_ax.minorticks_off()

            # Overlay zoomed-in plot.
            if TEMP_TYPE != "zoom":
                continue
            if axis != "rounds":
                continue
            zoom = 2
            axins.append(zoomed_inset_axes(
                current_ax,
                zoom,
                loc=LOC[metric],
                bbox_to_anchor=BBOX_TO_ANCHOR[metric],
                bbox_transform=current_ax.transAxes,
            ))
            for alg in ["local_clip", "episode_final"]:
                for I in [8, 16]:
                    candidate_keys = [key for key in keys if key.startswith(alg) and key.endswith(str(I))]
                    assert len(candidate_keys) == 1
                    key = candidate_keys[0]
                    if axis == "epochs":
                        x = [(i + 1) * max_points / len(results[metric][key]) for i in range(len(results[metric][key]))]
                    elif axis == "rounds":
                        total_rounds = total_updates // I
                        x = [(i+1) * total_rounds / len(results[metric][key]) for i in range(len(results[metric][key]))]
                    else:
                        raise NotImplementedError
                    axins[-1].plot(x, results[metric][key], color=COLORS[I], linestyle=STYLES[alg], markersize=MARKER_SIZE, linewidth=LINE_WIDTH)

            axins[-1].set_xscale("log")
            axins[-1].set_xlim(ZOOM_XLIM[metric])
            axins[-1].set_ylim(ZOOM_YLIM[metric])
            axins[-1].set_xticks([725, 2500])
            axins[-1].set_xticklabels([725, 2500])
            axins[-1].minorticks_off()
            corners = CORNERS[metric]
            mark_inset(current_ax, axins[-1], loc1=corners[0], loc2=corners[1], fc="none", ec="0.5", linestyle="--")

        fig.savefig(os.path.join(LOG_DIR, f"effect_of_I_{axis}.eps"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folders", nargs="+", type=str, help="Folders with logs.")
    parser.add_argument("--output_dir", type=str, default="../logs", help="Output location.")
    args = parser.parse_args()
    main(**vars(args))
