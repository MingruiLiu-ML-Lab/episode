"""
SNLI data loader.
"""

import os
import json
import pickle
import random
from math import ceil

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


GLOVE_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"
N_CLASSES = 3


class SNLIDataset(torch.utils.data.Dataset):

    def __init__(self, root="", split="train"):
        """ Initialize SNLI dataset. """

        assert split in ["train", "dev", "test"]
        self.root = os.path.join(root, "snli_1.0")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.n_classes = N_CLASSES

        """ Read and store data from files. """
        self.labels = ["entailment", "neutral", "contradiction"]
        labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Read sentence and label data for current split from files.
        s1_path = os.path.join(self.root, "SNLI", f"s1.{self.split}")
        s2_path = os.path.join(self.root, "SNLI", f"s2.{self.split}")
        target_path = os.path.join(self.root, "SNLI", f"labels.{self.split}")
        self.s1_sentences = [line.rstrip() for line in open(s1_path, "r")]
        self.s2_sentences = [line.rstrip() for line in open(s2_path, "r")]
        self.targets = np.array(
            [labels_to_idx[line.rstrip("\n")] for line in open(target_path, "r")]
        )
        assert len(self.s1_sentences) == len(self.s2_sentences)
        assert len(self.s1_sentences) == len(self.targets)
        self.dataset_size = len(self.s1_sentences)
        print(f"Loaded {self.dataset_size} sentence pairs for {self.split} split.")

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, "SNLI", VOCAB_NAME)
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "dev", "test"]:
                paths = [
                    os.path.join(self.root, "SNLI", f"s1.{split}"),
                    os.path.join(self.root, "SNLI", f"s2.{split}"),
                ]
                for path in paths:
                    for line in open(path, "r"):
                        for word in line.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, "GloVe", GLOVE_NAME)
        wordvec_path = os.path.join(self.root, "SNLI", WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    word, vec = line.split(' ', 1)
                    if word in vocab:
                        self.word_vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word_vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word_vec
        assert "</s>" in self.word_vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )
        for i in range(len(self.s2_sentences)):
            sent = self.s2_sentences[i]
            self.s2_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word_vec] +
                ["</s>"]
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """ Return a single element of the dataset. """

        # Encode sentences as sequence of glove vectors.
        sent1 = self.s1_sentences[idx]
        sent2 = self.s2_sentences[idx]
        s1_embed = np.zeros((len(sent1), GLOVE_DIM))
        s2_embed = np.zeros((len(sent2), GLOVE_DIM))
        for j in range(len(sent1)):
            s1_embed[j] = self.word_vec[sent1[j]]
        for j in range(len(sent2)):
            s2_embed[j] = self.word_vec[sent2[j]]
        s1_embed = torch.from_numpy(s1_embed).float()
        s2_embed = torch.from_numpy(s2_embed).float()

        # Convert targets to tensor.
        target = torch.tensor([self.targets[idx]]).long()

        return s1_embed, s2_embed, target

    @property
    def n_words(self):
        return len(self.word_vec)


def collate_pad(data_points):
    """ Pad data points with zeros to fit length of longest data point in batch. """

    s1_embeds = [x[0] for x in data_points]
    s2_embeds = [x[1] for x in data_points]
    targets = [x[2] for x in data_points]

    # Get sentences for batch and their lengths.
    s1_lens = np.array([sent.shape[0] for sent in s1_embeds])
    max_s1_len = np.max(s1_lens)
    s2_lens = np.array([sent.shape[0] for sent in s2_embeds])
    max_s2_len = np.max(s2_lens)
    lens = (s1_lens, s2_lens)

    # Encode sentences as glove vectors.
    bs = len(data_points)
    s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
    s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
    for i in range(bs):
        e1 = s1_embeds[i]
        e2 = s2_embeds[i]
        s1_embed[: len(e1), i] = e1.clone()
        s2_embed[: len(e2), i] = e2.clone()
    embeds = (
        torch.from_numpy(s1_embed).float(), torch.from_numpy(s2_embed).float()
    )

    # Convert targets to tensor.
    targets = torch.cat(targets)

    return embeds, lens, targets


def data_loader(
    dataset_name,
    root,
    batch_size,
    rank=0,
    world_size=1,
    heterogeneity=0.0,
    extra_bs=None,
    num_workers=1,
    small=False,
):
    """ Construct data loaders for training, validation, and test data. """

    if heterogeneity < 0:
        raise ValueError("Data heterogeneity must be positive.")
    if world_size == 1 and heterogeneity > 0:
        raise ValueError("Cannot create a heterogeneous dataset when world_size == 1.")

    train_set = SNLIDataset(root=root, split="train")
    val_set = SNLIDataset(root=root, split="dev")
    test_set = SNLIDataset(root=root, split="test")

    # Partition the training data into multiple clients. Data partitioning to create
    # heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if world_size > 1:
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

    # Collect indices of instances with each label.
    train_label_idxs = [
        (train_set.targets == i).nonzero()[0].tolist()
        for i in range(train_set.n_classes)
    ]
    for l in range(train_set.n_classes):
        random.shuffle(train_label_idxs[l])

    # Divide samples from each label into iid pool and non-iid pool. Note that samples
    # in iid pool are shuffled while samples in non-iid pool are sorted by label.
    iid_pool = []
    non_iid_pool = []
    for i in range(train_set.n_classes):
        iid_split = int((1.0 - heterogeneity) * len(train_label_idxs[i]))
        iid_pool += train_label_idxs[i][:iid_split]
        non_iid_pool += train_label_idxs[i][iid_split:]
    random.shuffle(iid_pool)

    # Allocate iid and non-iid samples to each worker.
    worker_train_idxs = [[] for _ in range(world_size)]
    num_iid = len(iid_pool) // world_size
    num_non_iid = len(non_iid_pool) // world_size
    partition_size = num_iid + num_non_iid
    for j in range(world_size):
        worker_train_idxs[j] += iid_pool[num_iid * j: num_iid * (j+1)]
        worker_train_idxs[j] += non_iid_pool[num_non_iid * j: num_non_iid * (j+1)]
        random.shuffle(worker_train_idxs[j])

    # Get indices of local validation and test dataset. Note that the validation and
    # test set are not split into `total_clients` partitions, just `world_size`
    # partitions, since evaluation does not need to happen separate for each client.
    # TODO: Do we really need to enforce that each partition of the test set has the
    # same size? We are just throwing away some test examples here and it may be
    # unnecessary.
    val_partition = len(val_set) // world_size
    test_partition = len(test_set) // world_size
    val_idxs = list(range(val_partition * world_size))
    test_idxs = list(range(test_partition * world_size))
    random.shuffle(val_idxs)
    random.shuffle(test_idxs)
    local_val_idxs = val_idxs[rank * val_partition: (rank+1) * val_partition]
    local_test_idxs = test_idxs[rank * test_partition: (rank+1) * test_partition]

    # Use miniature dataset, if necessary.
    if small:
        for r in range(world_size):
            worker_train_idxs[r] = worker_train_idxs[r][:round(len(worker_train_idxs[r]) / 100)]
        local_val_idxs = local_val_idxs[:round(len(local_val_idxs) / 100)]
        local_test_idxs = local_test_idxs[:round(len(local_test_idxs) / 100)]

    # Check that each worker training dataset is disjoint and that training set is
    # disjoint from validation set. This is slow, so only uncomment this for testing.
    """
    print("Testing partitioned dataset...")
    for i in range(world_size):
        current_idxs = worker_train_idxs[i]
        other_idxs = []
        for j in range(total_clients):
            if j == i:
                continue
            other_idxs += worker_train_idxs[j]
        for idx in current_idxs:
            assert idx not in other_idxs

    print("Testing passed!")
    """

    # Construct loaders for train, val, test, and extra sets.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(worker_train_idxs[rank]),
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(local_val_idxs),
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(local_test_idxs),
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=True,
    )

    extra_loader = None
    if extra_bs is not None:
        extra_loader = DataLoader(
            train_set,
            sampler=SubsetRandomSampler(worker_train_idxs[rank]),
            batch_size=extra_bs,
            num_workers=num_workers,
            collate_fn=collate_pad,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, extra_loader


def get_label_distribution(loader):
    """ Iterate over loader and return distribution of labels. """
    num_samples = torch.zeros(loader.n_classes)
    loader.reset()
    while loader.has_next():
        _, _, targets = loader.next_batch()
        for label in range(loader.n_classes):
            num_samples[label] += torch.sum(targets == label)
    num_samples /= loader.dataset_size
    return num_samples


def save_worker_idxs(train_loader, test_loader, rank):
    """ Save worker indexes of local dataset to file. """
    temp_path = f"worker_idxs_{rank}.pkl"
    worker_idxs = {"train": train_loader.local_idxs, "test": test_loader.local_idxs}
    with open(temp_path, "wb") as temp_file:
        pickle.dump(worker_idxs, temp_file)
