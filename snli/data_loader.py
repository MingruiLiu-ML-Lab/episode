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


GLOVE_URL = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"
N_CLASSES = 3


class SNLIDataLoader:
    """ Loading SNLI data for PyTorch training. """
    
    def __init__(self, batch_size, root, split="train", small=False, rank=0, world_size=1, heterogeneity=0.0):
        """ Init function for SNLIDataLoader. """

        assert split in ["train", "dev", "test"]
        if world_size == 1 and heterogeneity > 0:
            raise ValueError("Can't create heterogeneous dataset when world_size=1.")
        assert 0 <= rank <= world_size - 1

        self.batch_size = batch_size
        self.root = os.path.join(root, "snli_1.0")
        self.split = split
        self.small = small
        self.rank = rank
        self.world_size = world_size
        self.heterogeneity = heterogeneity
        self.embed_dim = GLOVE_DIM
        self.n_classes = N_CLASSES

        # Load dataset.
        self.load_data()
        if self.world_size > 1:
            self.partition()
            print(f"Keeping only partition {self.rank} of size {self.dataset_size}.")
        if self.small:
            self.dataset_size = self.dataset_size // 100
            self.s1_sentences = self.s1_sentences[:self.dataset_size]
            self.s2_sentences = self.s2_sentences[:self.dataset_size]
            self.targets = self.targets[:self.dataset_size]
            print(f"Using 1% of data: {self.dataset_size} pairs for {self.split}.")

        self._permutation = None
        self._next_pos = None

    def load_data(self):
        """ Read data from files and store for batching. """

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

    def reset(self):
        """
        Prepare for an iteration over the dataset: shuffle elements and reset counter
        over seen elements.
        """
        self._permutation = np.random.permutation(self.dataset_size)
        self._next_pos = 0
        self._step = 0

    def next_batch(self):
        """ Sample the next batch of elements from the dataset. """

        if not self.has_next():
            raise StopIteration

        # Get idxs of current batch.
        start = self._next_pos
        end = min(self._next_pos + self.batch_size, self.dataset_size)
        idxs = self._permutation[start:end]
        bs = end - start
        self._next_pos = end
        self._step += 1

        # Get sentences for batch and their lengths.
        s1_batch = [self.s1_sentences[idx] for idx in idxs]
        s1_lens = np.array([len(sent) for sent in s1_batch])
        max_s1_len = np.max(s1_lens)
        s2_batch = [self.s2_sentences[idx] for idx in idxs]
        s2_lens = np.array([len(sent) for sent in s2_batch])
        max_s2_len = np.max(s2_lens)
        lens = (s1_lens, s2_lens)

        # Encode sentences as glove vectors.
        s1_embed = np.zeros((max_s1_len, bs, GLOVE_DIM))
        s2_embed = np.zeros((max_s2_len, bs, GLOVE_DIM))
        for i in range(len(idxs)):
            for j in range(len(s1_batch[i])):
                s1_embed[j, i] = self.word_vec[s1_batch[i][j]]
            for j in range(len(s2_batch[i])):
                s2_embed[j, i] = self.word_vec[s2_batch[i][j]]
        embeds = (
            torch.from_numpy(s1_embed).float(), torch.from_numpy(s2_embed).float()
        )

        # Convert targets to tensor.
        targets = torch.from_numpy(self.targets[idxs]).long()

        return embeds, lens, targets

    def has_next(self):
        """ Return True until all batches have been seen. """
        return self._next_pos < self.dataset_size

    def partition(self):
        """
        Split the dataset into local datasets for each client and keep only the local
        dataset with index `self.rank`.
        """

        # Fix random seed so that data is shuffled the same across clients.
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)

        # Collect indices of instances with each label.
        train_label_idxs = [
            (self.targets == i).nonzero()[0].tolist()
            for i in range(self.n_classes)
        ]
        label_proportions = torch.tensor(
            [float(len(train_label_idxs[i])) for i in range(self.n_classes)]
        )
        label_proportions /= torch.sum(label_proportions)
        for l in range(self.n_classes):
            random.shuffle(train_label_idxs[l])

        # Divide samples from each label into iid pool and non-iid pool. Note that samples
        # in iid pool are shuffled while samples in non-iid pool are sorted by label.
        iid_pool = []
        non_iid_pool = []
        for i in range(self.n_classes):
            iid_split = int((1.0 - self.heterogeneity) * len(train_label_idxs[i]))
            iid_pool += train_label_idxs[i][:iid_split]
            non_iid_pool += train_label_idxs[i][iid_split:]
        random.shuffle(iid_pool)

        # Allocate iid and non-iid samples to each worker.
        iid_start = 0
        non_iid_start = 0
        partition_size = self.dataset_size // self.world_size
        worker_idxs = [[] for _ in range(self.world_size)]
        for j in range(self.world_size):
            num_iid = int((1.0 - self.heterogeneity) * partition_size)
            num_non_iid = partition_size - num_iid
            worker_idxs[j] += iid_pool[iid_start: iid_start + num_iid]
            worker_idxs[j] += non_iid_pool[non_iid_start: non_iid_start + num_non_iid]
            iid_start += num_iid
            non_iid_start += num_non_iid
            random.shuffle(worker_idxs[j])

        # Keep only dataset elements for current worker.
        self.local_idxs = list(worker_idxs[self.rank])
        self.s1_sentences = [self.s1_sentences[idx] for idx in self.local_idxs]
        self.s2_sentences = [self.s2_sentences[idx] for idx in self.local_idxs]
        self.targets = self.targets[self.local_idxs]
        self.dataset_size = len(self.local_idxs)

    def state(self):
        """ State of the data loader. """
        state = {
            "permutation": self._permutation.copy(),
            "next_pos": self._next_pos,
            "step": self._step,
        }
        return state

    def load_state(self, state):
        """ Load a state for the data loader. """
        self._permutation = state["permutation"]
        self._next_pos = state["next_pos"]
        self._step = state["step"]

    @property
    def n_words(self):
        return len(self.word_vec)

    @property
    def num_batches(self):
        return ceil(self.dataset_size / self.batch_size)


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
