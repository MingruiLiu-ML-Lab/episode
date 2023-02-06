""" Plot training results from tensorboard files. """

import os
import argparse
import glob
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LABELS = {
    "train/loss": "Train Loss",
    "train/accuracy": "Train Accuracy",
    "val/loss": "Test Loss",
    "val/accuracy": "Test Accuracy",
    "scaffold": "SCAFFOLD",
    "episode": "EPISODE",
    "fedavg": "FedAvg",
    "local_clip": "CELGC",
}
YLIM = {
    "loss": [0.8, 3.0],
    "accuracy": [0.4, 0.8]
}
ZOOM_XLIM = {
    "train/loss": [85, 90],
    "val/accuracy": [85, 90]
}
ZOOM_YLIM = {
    "train/loss": [0.9, 1.0],
    "val/accuracy": [0.735, 0.755]
}
BBOX_TO_ANCHOR = {
    "train/loss": (1.0, 0.75, 0, 0),
    "val/accuracy": (0.65, 0.83, 0, 0),
}
LOC = {
    "train/loss": 1,
    "val/accuracy": 2,
}
CORNERS = {
    "train/loss": (3, 4),
    "val/accuracy": (1, 2),
}


plt.rcParams.update({'font.size': 14, 'font.family': "FreeSerif"})


def read_from_tfevents_log(event_file_path):
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()
    results = {}
    for scalar_tag in event_acc.Tags()['scalars']:
        results[scalar_tag] = np.array(
            [x.value for x in event_acc.Scalars(scalar_tag)]
        )
    results['train/clip_ratio'] = np.divide(
        results['train/clipites'],
        results['train/totalites'],
        out=np.zeros_like(results['train/clipites']),
        where=(results['train/totalites']!=0)
    )
    return results


def main(log_folder, epochs):

    # Read in worker results from file.
    worker_results = {}
    method_dirs = next(os.walk(log_folder))[1]
    for method in method_dirs:
        worker_results[method] = {}
        for rank_dir in os.listdir(os.path.join(log_folder, method)):
            rank = int(rank_dir[rank_dir.index('rank_') + len('rank_'):])
            event_path = glob.glob(os.path.join(log_folder, method, rank_dir, "events.out.tfevents.*"))
            assert len(event_path) == 1
            event_path = event_path[0]
            worker_results[method][rank] = read_from_tfevents_log(event_path)

    # Re-order methods.
    methods = list(worker_results.keys())
    if set(methods) == set(["scaffold", "local_clip", "episode", "fedavg"]):
        methods = ["fedavg", "local_clip", "scaffold", "episode"]

    # Average results over workers.
    results = {}
    for method in methods:
        results[method] = {}
        for metric in worker_results[method][0].keys():
            results[method][metric] = np.mean(
                np.stack(
                    [
                        worker_results[method][rank][metric]
                        for rank in worker_results[method].keys()
                    ],
                    axis=0
                ),
                axis=0
            )

    """
    print(len(results))
    print(list(results.keys()))
    for method in results.keys():
        print((results[method].keys()))
        for metric in results[method].keys():
            print(results[method][metric].shape)
    """

    # Plot results.
    for metric in ["loss", "accuracy"]:
        plt.clf()
        fig, axs = plt.subplots(2, 1, figsize=[5, 9])
        for i, split in enumerate(["train", "val"]):
            full_metric = f"{split}/{metric}"
            for method in methods:
                axs[i].plot(results[method][full_metric], label=LABELS[method])
                print(f"{full_metric} {method}: {results[method][full_metric][-1]}")
            axs[i].set_ylim(YLIM[metric])
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(LABELS[full_metric])
            axs[i].legend()

        plot_path = os.path.join(log_folder, f"{metric}.eps")
        plt.savefig(plot_path)

    plt.clf()
    fig = plt.gcf()
    for method in results:
        plt.plot(results[method]["train/clip_ratio"], label=LABELS[method])
    plt.xlabel("Epoch")
    plt.ylabel("Clip Frequency")
    plt.legend()
    plot_path = os.path.join(log_folder, "clip_frequency.eps")
    plt.savefig(plot_path)

    # Make final plot to appear in paper.
    plt.clf()
    final_metrics = [("train", "loss"), ("val", "accuracy")]
    fig, axs = plt.subplots(len(final_metrics), 1, figsize=[5,9])
    axins = []
    for i, (split, metric) in enumerate(final_metrics):
        full_metric = f"{split}/{metric}"
        for method in methods:
            axs[i].plot(results[method][full_metric], label=LABELS[method])
        axs[i].set_ylim(YLIM[metric])
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel(LABELS[full_metric])
        if i == len(final_metrics) - 1:
            axs[i].legend()

        # Overlay zoomed-in plot.
        zoom = 6
        axins.append(zoomed_inset_axes(
            axs[i],
            zoom,
            loc=LOC[full_metric],
            bbox_to_anchor=BBOX_TO_ANCHOR[full_metric],
            bbox_transform=axs[i].transAxes,
        ))
        for method in methods:
            axins[i].plot(results[method][full_metric], label=LABELS[method])

        # Adjust zoom window.
        ends = [results[method][full_metric][-1] for method in methods]
        y_lim = ZOOM_YLIM[full_metric]
        all_in = all([end > y_lim[0] and end < y_lim[1] for end in ends])
        if not all_in:
            window = max(ends) - min(ends)
            cushion = window * 0.25
            y_lim = [min(ends) - cushion, max(ends) + cushion]

        axins[i].set_xlim(ZOOM_XLIM[full_metric])
        axins[i].set_ylim(y_lim)
        axins[i].set_xticks([86, 89])
        axins[i].set_xticklabels(["86", "89"])
        corners = CORNERS[full_metric]
        mark_inset(axs[i], axins[i], loc1=corners[0], loc2=corners[1], fc="none", ec="0.5", linestyle="--")

    plot_path = os.path.join(log_folder, f"final.eps")
    plt.savefig(plot_path, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder", help="Folder containing results to plot",
                        type=str)
    parser.add_argument("--epochs", help="maximum number of epochs to draw", type=int,
                        default=90)
    args = parser.parse_args()
    main(**vars(args))
