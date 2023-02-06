import os
import argparse
import glob
import json

import numpy as np


def main(log_folder):

    # Get worker logs.
    client_results = []
    worker_logs = glob.glob(os.path.join(log_folder, "*.json"))
    for worker_log in worker_logs:
        with open(worker_log, "r") as f:
            client_results.append(json.load(f))
    keys = list(client_results[0].keys())
    for client_result in client_results[1:]:
        assert keys == list(client_result.keys())

    # Average results.
    avg_results = {}
    for key in keys:
        if key in ["eval_clip_operations", "eval_corrected_operations"]:
            continue
        avg_results[key] = np.mean(
            [client_result[key] for client_result in client_results],
            axis=0
        ).tolist()
    log_name = worker_logs[0]
    rank_pos = log_name.rfind("Rank")
    log_name = log_name[:rank_pos-1] + ".json"
    with open(log_name, 'w') as f:
        json.dump(avg_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_folders", nargs="+", help="Folder containing results to plot", type=str
    )
    args = parser.parse_args()
    for log_folder in args.log_folders:
        main(log_folder)
