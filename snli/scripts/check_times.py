import os
import argparse
import json

import numpy as np


TIME_METRIC = "eval_elasped_times"
PERF_METRIC = "test_accuracies"
BENCHMARKS = [0.7, 0.75, 0.8]


def main(log_folder):

    # Read results of each algorithm (averaged over workers).
    results = {}
    steps = None
    for alg_path, _, alg_files in list(os.walk(log_folder))[1:]:
        alg = os.path.basename(alg_path)
        candidate_avg = [filename for filename in alg_files if "Rank" not in filename and filename.endswith(".json")]
        if len(candidate_avg) != 1:
            print(f"Incomplete results for {alg}, skipping.")
            continue
        avg_filename = candidate_avg[0]
        with open(os.path.join(alg_path, avg_filename), "r") as f:
            results[alg] = json.load(f)

        assert TIME_METRIC in results[alg]
        assert PERF_METRIC in results[alg]
        assert len(results[alg][TIME_METRIC]) == len(results[alg][TIME_METRIC])
        if steps is None:
            steps = len(results[alg][TIME_METRIC])
        else:
            assert steps == len(results[alg][TIME_METRIC])

    # Check first time for each accuracy benchmark.
    firsts = {}
    for alg in results:
        firsts[alg] = {}
        for benchmark in BENCHMARKS:
            first_t = 0
            while first_t < steps:
                if results[alg][PERF_METRIC][first_t] >= benchmark:
                    firsts[alg][benchmark] = sum(results[alg][TIME_METRIC][:first_t+1]) / 60
                    break
                first_t += 1
            if benchmark not in firsts[alg]:
                firsts[alg][benchmark] = None

    # Print firsts.
    for alg in results:
        msg = f"{alg} {BENCHMARKS}:"
        for benchmark in BENCHMARKS:
            t = firsts[alg][benchmark]
            if t is not None:
                msg += f" {firsts[alg][benchmark]:.2f}"
            else:
                msg += " None"
        print(msg)
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_folders", nargs="+", help="Folder containing results to plot", type=str
    )
    args = parser.parse_args()
    for log_folder in args.log_folders:
        print(log_folder)
        main(log_folder)
