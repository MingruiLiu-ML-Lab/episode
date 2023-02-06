""" Plot training results. """

import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = "../logs"
LOG_TEMPLATES = ["SNLI_SGDClipGrad_Eta0_0.*"]
ALGS = ["local_clip", "episode_final"]


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

    # Find best configuration for each algorithm.
    best_keys = {}
    for alg in ALGS:
        alg_keys = [key for key in results["train_losses"].keys() if key.startswith(alg)]
        alg_losses = {key: results["train_losses"][key][-1] for key in alg_keys}
        best_keys[alg] = None
        best_loss = None
        for key, loss in alg_losses.items():
            if best_loss is None or loss < best_loss:
                best_keys[alg] = key
                best_loss = float(loss)
        print(f"{alg}: {best_keys[alg]}, {best_loss}")

    # Collect best results and maximum number of points to plot across algorithms.
    best_results = {}
    max_points = {"train_losses": 0, "test_losses": 0, "train_accs": 0, "test_accs": 0, "clipping": 0}
    for metric in ["train_losses", "test_losses", "train_accs", "test_accs", "clipping"]:
        best_results[metric] = {}
        for alg in ALGS:
            best_results[metric][alg] = results[metric][best_keys[alg]]
            max_points[metric] = max(max_points[metric], len(best_results[metric][alg]))
            if metric == "clipping":
                print(f"{alg} min, max: {np.min(best_results[metric][alg])} {np.max(best_results[metric][alg])}")
    
    # Plot results.
    for metric in ["train_losses", "test_losses", "train_accs", "test_accs", "clipping"]:
        plt.clf()
        for alg in ALGS:
            x = [(i + 1) * max_points[metric] / len(best_results[metric][alg]) for i in range(len(best_results[metric][alg]))]
            plt.plot(x, best_results[metric][alg], label=alg)
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, f"{metric}_plot.svg"))


if __name__ == "__main__":
    main()
