import os
import random
import json
import argparse

import numpy as np


DATA_ROOT = "../data/sent140"
TRAIN_NAME = "train.json"
TEST_NAME = "test.json"


def process_label(l):
    if l == "0":
        return 0
    elif l == "4":
        return 1
    else:
        raise ValueError


def main(world_size, silo_hetero):
    """ Split pool of clients into silos. """

    assert 0 <= silo_hetero <= 1

    # Read train data and test data.
    with open(os.path.join(DATA_ROOT, TRAIN_NAME), "r") as f:
        train_dset = json.load(f)
    with open(os.path.join(DATA_ROOT, TEST_NAME), "r") as f:
        test_dset = json.load(f)

    # Make sure train users are ordered the same as test users.
    train_users = train_dset["users"]
    test_users = test_dset["users"]
    assert train_users == test_users
    users = list(train_users)
    num_users = len(users)

    # Get label distribution of each client.
    label_dist = []
    for i, user in enumerate(users):
        user_labels = train_dset["user_data"][user]["y"] + test_dset["user_data"][user]["y"]
        user_labels = [process_label(l) for l in user_labels]
        label_dist.append(np.mean(user_labels))

    # Split clients into iid pool and non-iid pool, and sort non-iid pool by label dist.
    client_idxs = list(range(num_users))
    random.shuffle(client_idxs)
    num_noniid = round(num_users * silo_hetero)
    num_iid = num_users - num_noniid
    iid_pool = np.array(client_idxs[:num_iid])
    noniid_pool = np.array(client_idxs[num_iid:])
    noniid_dist = [label_dist[i] for i in noniid_pool]
    sorted_idxs = np.argsort(noniid_dist)
    noniid_pool = noniid_pool[sorted_idxs]

    # Allocate clients from iid pool and non-iid pool to each silo.
    silo_clients = []
    iid_p_size = len(iid_pool) / world_size
    noniid_p_size = len(noniid_pool) / world_size
    for i in range(world_size):
        iid_start = round(iid_p_size * i)
        iid_end = round(iid_p_size * (i+1))
        iid_clients = iid_pool[iid_start:iid_end].tolist()

        noniid_start = round(noniid_p_size * i)
        noniid_end = round(noniid_p_size * (i+1))
        noniid_clients = noniid_pool[noniid_start:noniid_end].tolist()

        silo_clients.append(iid_clients + noniid_clients)

    # Compute label distribution of each silo.
    for i in range(world_size):
        silo_labels = []
        for client in silo_clients[i]:
            user = users[client]
            silo_labels += train_dset["user_data"][user]["y"] + test_dset["user_data"][user]["y"]
        silo_labels = [process_label(l) for l in silo_labels]
        print(len(silo_labels))
        silo_dist = np.mean(silo_labels)
        print(f"Silo {i}: {silo_dist}")

    # Write silos to file.
    silo_clients_path = os.path.join(DATA_ROOT, "silos", f"silo_{world_size}_{silo_hetero}.json")
    if not os.path.isdir(os.path.dirname(silo_clients_path)):
        os.makedirs(os.path.dirname(silo_clients_path))
    with open(silo_clients_path, "w") as f:
        json.dump(silo_clients, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("world_size", type=int)
    parser.add_argument("silo_hetero", type=float)
    args = parser.parse_args()
    main(**vars(args))
