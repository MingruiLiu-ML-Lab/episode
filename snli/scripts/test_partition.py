import pickle
import random


WORLD_SIZE = 4


worker_idxs = {"train": {}, "test": {}}
for rank in range(WORLD_SIZE):
    rank_path = f"worker_idxs_{rank}.pkl"
    with open(rank_path, "rb") as rank_file:
        rank_idxs = pickle.load(rank_file)
    worker_idxs["train"][rank] = sorted(rank_idxs["train"])
    worker_idxs["test"][rank] = sorted(rank_idxs["test"])

for rank in range(WORLD_SIZE):
    for split in ["train", "test"]:
        current = worker_idxs[split][rank]
        other = []
        for other_rank in range(WORLD_SIZE):
            if other_rank == rank:
                continue
            other += worker_idxs[split][other_rank]
        other = sorted(other)
        current_pos = 0
        other_pos = 0
        while current_pos < len(current) and other_pos < len(other):
            if current[current_pos] < other[other_pos]:
                current_pos += 1
            elif current[current_pos] > other[other_pos]:
                other_pos += 1
            else:
                assert False

print("passed! worker datasets are disjoint")
