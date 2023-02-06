""" Plot training results. """

import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = "../logs"
LOG_TEMPLATES = ["SNLI_SGDClipGrad_Eta0_0.*"]

COLORS = {
    1: "red",
    2: "orange",
    4: "yellow",
    8: "green",
    16: "blue",
}
STYLES = {
    "local_clip": "solid",
    "episode_final": "dashed",
    "single_clip": "dotted",
}


def main():

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
        """
        bs_scale_pos = aggregate_name.find("BSScale")
        if bs_scale_pos == -1:
            bs_scale = None
        else:
            under_pos_1 = aggregate_name.find("_", bs_scale_pos + 1)
            under_pos_2 = aggregate_name.find("_", under_pos_1 + 1)
            bs_scale = float(aggregate_name[under_pos_1 + 1: under_pos_2])
        if bs_scale is not None:
            key += f"_{bs_scale}"
        """
        """
        under_pos_1 = aggregate_name.find("_I_") + 2
        under_pos_2 = aggregate_name.find("_", under_pos_1 + 1)
        I = aggregate_name[under_pos_1+1: under_pos_2]
        key += f"_{I}"
        """

        # Read results for each worker and average them.
        worker_names = glob.glob(os.path.join(LOG_DIR, aggregate_name + "*"))
        worker_train_losses = []
        worker_test_losses = []
        worker_train_accs = []
        worker_test_accs = []
        worker_clipping = []
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

        results["train_losses"][key] = np.mean(np.array(worker_train_losses), axis=0)
        results["test_losses"][key] = np.mean(np.array(worker_test_losses), axis=0)
        results["train_accs"][key] = np.mean(np.array(worker_train_accs), axis=0)
        results["test_accs"][key] = np.mean(np.array(worker_test_accs), axis=0)
        results["clipping"][key] = np.mean(np.array(worker_clipping), axis=0)

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
            if metric == "clipping":
                print(f"{key} min, max: {np.min(results[metric][key])} {np.max(results[metric][key])}")
            if metric == "test_losses":
                print(f"{key} test_loss: {results[metric][key][-1]}")

    # Plot results.
    for metric in ["train_losses", "test_losses", "train_accs", "test_accs", "clipping"]:
        plt.clf()
        plt.figure(figsize=(16,16))
        for key in keys:
            x = [(i + 1) * max_points[metric] / len(results[metric][key]) for i in range(len(results[metric][key]))]
            plt.plot(x, results[metric][key], label=key)
        """
        for I in [1, 2, 4, 8, 16]:
            for alg in ["local_clip", "episode_final"]:
                key = f"{alg}_0.1_0.03_{I}"
                x = [(i + 1) * max_points[metric] / len(results[metric][key]) for i in range(len(results[metric][key]))]
                plt.plot(x, results[metric][key], label=key, color=COLORS[I], linestyle=STYLES[alg])
        alg = "single_clip"
        I = 1
        key = f"{alg}_0.1_0.03_128"
        x = [(i + 1) * max_points[metric] / len(results[metric][key]) for i in range(len(results[metric][key]))]
        plt.plot(x, results[metric][key], label=key, color=COLORS[I], linestyle=STYLES[alg])
        """
        plt.legend()
        plt.ylim([0, 1.0])
        plt.savefig(os.path.join(LOG_DIR, f"{metric}_plot.svg"))


if __name__ == "__main__":
    main()
