"""
Create train, valid, test iterators for a chosen dataset.
"""

import torch
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def data_loader(dataset_name, dataroot, batch_size, val_ratio, world_size, rank, heterogeneity=0, extra_bs=None, num_workers=1, small=False):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroor (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation.
        world_size (int): how many processed will be used in training.
        rank (int): the rank of this process.
        heterogeneity (float): dissimilarity between data distribution across clients.
            Between 0 and 1.
        extra_bs (int): Batch size for extra data loader.
        small (bool): Whether to use miniature dataset.

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
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
        num_labels = 10
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
        num_labels = 100
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        num_labels = 10
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        num_labels = 10
    else:
        raise NotImplementedError

    if dataset_name.startswith('CIFAR'):
        # Follows Lee et al. Deeply supervised nets. 2014.
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train

    # load and split the train dataset into train and validation and 
    # deployed to all GPUs.
    train_set = dataset(root=dataroot, train=True,
                        download=True, transform=transform_train)
    val_set = dataset(root=dataroot, train=True,
                      download=True, transform=transform_test)
    test_set = dataset(root=dataroot, train=False,
                       download=True, transform=transform_test)

    # partition the training data into multiple GPUs if needed. Data partitioning to
    # create heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if world_size > 1:
        random.seed(1234)
    train_data_len = len(train_set)
    train_label_idxs = get_label_indices(dataset_name, train_set, num_labels)
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

    # Get indices of local test dataset.
    test_partition = len(test_set) // world_size
    test_idxs = list(range(test_partition * world_size))
    random.shuffle(test_idxs)
    local_test_idx = test_idxs[rank * test_partition: (rank+1) * test_partition]

    # Use miniature dataset, if necessary.
    if small:
        local_train_idx = local_train_idx[:round(len(local_train_idx) / 100)]
        local_valid_idx = local_valid_idx[:round(len(local_valid_idx) / 100)]
        local_test_idx = local_test_idx[:round(len(local_test_idx) / 100)]

    # Construct loaders for train, valid, extra, and test sets.
    train_sampler = SubsetRandomSampler(local_train_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_sampler = SubsetRandomSampler(local_valid_idx)
    valid_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    extra_loader = None
    if extra_bs is not None:
        extra_loader = DataLoader(
            train_set,
            batch_size=extra_bs,
            sampler=SubsetRandomSampler(local_train_idx),
            num_workers=num_workers,
            pin_memory=True,
        )

    test_sampler = SubsetRandomSampler(local_test_idx)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (train_loader, valid_loader, test_loader, extra_loader)


def get_label_indices(dataset_name, dset, num_labels):
    """
    Returns a dictionary mapping each label to a list of the indices of elements in
    `dset` with the corresponding label.
    """

    label_indices = [[] for _ in range(num_labels)]
    if dataset_name in ["CIFAR10", "CIFAR100", "MNIST"]:
        for idx, label in enumerate(dset.targets):
            label_indices[label].append(idx)
    else:
        raise NotImplementedError

    return label_indices
