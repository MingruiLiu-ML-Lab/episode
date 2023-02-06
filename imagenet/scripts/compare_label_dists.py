import torch

WORLD_SIZE = 8
H = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
SPLITS = ["train", "val"]
#SPLITS = ["val"]

for split in SPLITS:
    for hetero in H:
        dists = [torch.load(f"label_dist_{split}_{hetero}_{rank}.pth") for rank in range(WORLD_SIZE)]

        similarity = torch.zeros(WORLD_SIZE, WORLD_SIZE)
        for rank1 in range(WORLD_SIZE):
            for rank2 in range(WORLD_SIZE):
                similarity[rank1, rank2] = torch.distributions.kl.kl_divergence(
                    torch.distributions.categorical.Categorical(dists[rank1]),
                    torch.distributions.categorical.Categorical(dists[rank2]),
                )

        print(hetero, split)
        print(similarity)
        print("")
