""" Plot training results. """

import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


LINE_WIDTH = 1.5
TITLES = {
    "train_losses": "Train loss",
    "train_accuracies": "Train Accuracy",
    "test_losses": "Test accuracy",
    "test_accuracies": "Test accuracy",
    "epoch_clip_operations": "Clipping Frequency",
    "rounds": "Round",
    "local_clip": "CELGC",
    "episode_final": "EPISODE",
    "fedavg": "FedAvg",
    "SCAFFOLD": "SCAFFOLD",
    "global_avg_clip": "NaiveParallelClip",
    "scaffold_clip": "SCAFFOLD clipped",
    "episode_no_clip": "EPISODE unclipped"
}
ALG_ORDER = ["global_avg_clip", "local_clip", "episode_final", "scaffold_clip", "episode_no_clip", "fedavg", "SCAFFOLD"]


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

        if "eval_clip_operations" in results[alg]:
            results[alg]["eval_clip_operations"] = np.mean(results[alg]["eval_clip_operations"], axis=1)

    unwanted_metrics = ["eval_corrected_operations"]
    for m in unwanted_metrics:
        if m in metrics:
            metrics.remove(m)

    # Reorder algorithms.
    methods = list(results.keys())
    methods = sorted(methods, key=(lambda alg: ALG_ORDER.index(alg)))

    # Plot results.
    for metric in ["train_losses", "test_accuracies"]:
        plt.clf()
        for alg in methods:
            if alg in ["fedavg", "SCAFFOLD", "episode_no_clip"] and metric == "train_losses":
                continue
            x = np.arange(len(results[alg][metric]))
            plt.plot(x, results[alg][metric], label=TITLES[alg], linewidth=LINE_WIDTH)
        plt.xlabel("Epochs")
        plt.ylabel(TITLES[metric])
        plt.legend()

        if metric == "train_losses":
            plt.ylim([0.3, 1.0])
        elif metric == "test_accuracies":
            plt.ylim([0.3, 0.85])
        else:
            raise NotImplementedError

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
        print("")
