"""
Create train, valid, test iterators for a chosen dataset.
"""

import os
import random
import torch
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def data_loader(dataroot, batch_size, val_ratio, world_size, rank, heterogeneity=0, extra_bs=None, small=False, num_workers=1):
    """
    Args:
        dataroot (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation.
        world_size (int): how many processed will be used in training.
        rank (int): the rank of this process.
        heterogeneity (float): dissimilarity between data distribution across clients.
            Between 0 and 1.
        extra_bs (int): Batch size for extra data loader. Only used for EPISODE.

    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")
    if heterogeneity < 0:
        raise ValueError("Data heterogeneity must be positive.")
    if world_size == 1 and heterogeneity > 0:
        raise ValueError("Cannot create a heterogeneous dataset when world_size == 1.")

    # Mean and std are obtained for each channel from all training images.
    dataset = torchvision.datasets.ImageFolder
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    num_labels = 1000
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # load and split the train dataset into train and validation and 
    # deployed to all GPUs.
    if small:
        dataroot = os.path.join(os.path.dirname(dataroot), "small_imagenet")
    train_set = dataset(root=os.path.join(dataroot, "train"), transform=transform_train)
    val_set = dataset(root=os.path.join(dataroot, "train"), transform=transform_test)
    test_set = dataset(root=os.path.join(dataroot, "val"), transform=transform_test)

    # partition the training data into multiple GPUs if needed. Data partitioning to
    # create heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if world_size > 1:
        random.seed(1234)
    train_data_len = len(train_set)
    train_label_idxs = get_label_indices(train_set, num_labels)
    label_proportions = torch.tensor([float(len(train_label_idxs[i])) for i in range(num_labels)])
    label_proportions /= torch.sum(label_proportions)
    for l in range(num_labels):
        random.shuffle(train_label_idxs[l])
    worker_idxs = [[] for _ in range(world_size)]

    # Divide samples from each label into iid pool and non-iid pool. Note that samples
    # in iid pool are shuffled while samples in non-iid pool are sorted by label.
    iid_pool = []
    non_iid_pool = []
    for i in range(num_labels):
        iid_split = round((1.0 - heterogeneity) * len(train_label_idxs[i]))
        iid_pool += train_label_idxs[i][:iid_split]
        non_iid_pool += train_label_idxs[i][iid_split:]
    random.shuffle(iid_pool)

    # Allocate iid and non-iid samples to each worker.
    num_iid = len(iid_pool) // world_size
    num_non_iid = len(non_iid_pool) // world_size
    partition_size = num_iid + num_non_iid
    for j in range(world_size):
        worker_idxs[j] += iid_pool[num_iid * j: num_iid * (j+1)]
        worker_idxs[j] += non_iid_pool[num_non_iid * j: num_non_iid * (j+1)]
        random.shuffle(worker_idxs[j])

    # Split training set into training and validation for current worker.
    val_split = int(val_ratio * partition_size)
    local_train_idx = worker_idxs[rank][val_split:]
    local_valid_idx = worker_idxs[rank][:val_split]

    # Check that each worker dataset is disjoint. This is slow, so only comment this out
    # for testing.
    """
    for i in range(world_size):
        current_idxs = worker_idxs[i]
        other_idxs = []
        for j in range(world_size):
            if j == i:
                continue
            other_idxs += worker_idxs[j]
        for idx in current_idxs:
            assert idx not in other_idxs
    """

    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to
    # prevent issues with Infiniband implementations that are not fork-safe
    kwargs = {}
    if (
        hasattr(mp, '_supports_context') and
        mp._supports_context and
        'forkserver' in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = 'forkserver'

    # Construct samplers for train and valid sets.
    train_sampler = SubsetRandomSampler(local_train_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    valid_sampler = SubsetRandomSampler(local_valid_idx)
    valid_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    # Construct extra data loaders, if necessary.
    extra_loader = None
    if extra_bs is not None:
        extra_loader = DataLoader(
            train_set,
            batch_size=extra_bs,
            sampler=SubsetRandomSampler(local_train_idx),
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )

    # Load the test dataset.
    test_partition = len(test_set) // world_size
    test_idxs = list(range(test_partition * world_size))
    random.shuffle(test_idxs)
    local_test_idx = test_idxs[rank * test_partition: (rank+1) * test_partition]
    test_sampler = SubsetRandomSampler(local_test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )

    return (train_loader, valid_loader, test_loader, extra_loader)


def get_label_indices(dset, num_labels):
    """
    Returns a dictionary mapping each label to a list of the indices of elements in
    `dset` with the corresponding label.
    """
    label_indices = [[] for _ in range(num_labels)]
    for idx, (_, label) in enumerate(dset.samples):
        label_indices[label].append(idx)
    return label_indices


def get_label_distribution(loader, num_labels):
    """
    Debug function to test the distribution of labels in a given dataset.
    """

    label_dist = torch.zeros(num_labels).cuda()
    total = 0
    for _, target in loader:
        target = target.cuda()
        total += len(target)
        label_dist += torch.bincount(target, minlength=num_labels)
    label_dist /= total
    return label_dist.cpu()


def save_label_distributions(train_loader, val_loader, num_labels):
    """
    Debug function to save distribution of labels to file.
    """
    train_label_dist = get_label_distribution(train_loader, 1000)
    val_label_dist = get_label_distribution(val_loader, 1000)
    torch.save(train_label_dist, f"label_dist_train_{args.heterogeneity}_{hvd.rank()}.pth")
    torch.save(val_label_dist, f"label_dist_val_{args.heterogeneity}_{hvd.rank()}.pth")
