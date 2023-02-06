"""
Sent140 data loader.
"""

import os
import json
import pickle
import random
from math import ceil
from tqdm import tqdm

import numpy as np
import torch


GLOVE_NAME = "glove.840B.300d.txt"
GLOVE_DIM = 300
VOCAB_NAME = "vocab.pkl"
WORDVEC_NAME = "wordvec.pkl"
N_CLASSES = 2


class Sent140DataLoader:
    """ Loads Sent140 data for PyTorch training. """

    def __init__(
        self, batch_size, root, rank, world_size, split="train", small=False, client_sampling="random", silo_hetero=0.0
    ):
        """ Init function for Sent140DataLoader. """

        assert split in ["train", "test"]

        self.batch_size = batch_size
        self.root = os.path.join(root, "sent140")
        self.data_path = os.path.join(self.root, f"{split}.json")
        self.rank = rank
        self.world_size = world_size
        self.split = split
        self.small = small
        self.client_sampling = client_sampling
        self.silo_hetero = silo_hetero
        self.embed_dim = GLOVE_DIM
        self.n_classes = N_CLASSES

        self.load_data()

        self._clients = None
        self._current_sentences = None
        self._current_labels = None
        self.dataset_size = None
        self._permutation = None
        self._next_pos = None
        self._step = None

    def load_data(self):
        """ Read data from files and store for batching. """

        # Read sentence and label data for current split from file.
        with open(self.data_path, "r") as f:
            all_data = json.load(f)
        self.users = range(len(all_data["users"]))
        self.num_users = len(self.users)
        if self.small:
            self.num_users = self.num_users // 4
        self.user_sizes = {}
        self.sentences = {}
        self.labels = {}
        def process_label(l):
            if l == "0":
                return 0
            elif l == "4":
                return 1
            else:
                raise ValueError

        for i in self.users:
            user = all_data["users"][i]
            self.user_sizes[i] = 0
            self.sentences[i] = []
            self.labels[i] = []
            tweets = all_data["user_data"][user]["x"]
            labels = all_data["user_data"][user]["y"]
            assert len(tweets) == len(labels)
            for tweet_data, label in zip(tweets, labels):
                self.sentences[i].append(tweet_data[4])
                self.labels[i].append(process_label(label))
                self.user_sizes[i] += 1

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = os.path.join(self.root, VOCAB_NAME)
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
            for split in ["train", "test"]:
                path = os.path.join(self.root, f"{split}.json")
                with open(path, "r") as f:
                    split_data = json.load(f)
                split_sentences = split_data["user_data"]

                for user in tqdm(split_data["users"]):
                    for tweet_data in split_data["user_data"][user]["x"]:
                        sentence = tweet_data[4]
                        for word in sentence.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word_vec = {}
        glove_path = os.path.join(self.root, GLOVE_NAME)
        wordvec_path = os.path.join(self.root, WORDVEC_NAME)
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word_vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            lines = []
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    lines.append(line)
            for line in tqdm(lines):
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
        for user in self.sentences:
            for i, sentence in enumerate(self.sentences[user]):
                self.sentences[user][i] = np.array(
                    ["<s>"] +
                    [word for word in sentence.split() if word in self.word_vec] +
                    ["</s>"]
                )

        # If necessary, read clients in current silo from file. Note that there is a
        # separate script (scripts/silo.py) to split all clients into silos to ensure
        # that this split is the same on all workers.
        if self.client_sampling == "silo":

            silo_clients_path = os.path.join(
                self.root, "silos", f"silo_{self.world_size}_{self.silo_hetero}.json"
            )
            if not os.path.isfile(silo_clients_path):
                raise ValueError(
                    f"File {silo_clients_path} holding partition of clients into silos"
                    " does not exist. Run scripts/silo.py to generate it before"
                    " running training."
                )
            with open(silo_clients_path, "r") as f:
                silo_clients = json.load(f)
            self.silo_clients = silo_clients[self.rank]

    def set_clients(self, clients):
        """ Change client(s) to load data from. """
        for client in clients:
            assert 0 <= client < self.num_users
        self._clients = list(clients)
        self._current_sentences = []
        self._current_labels = []
        for client in self._clients:
            self._current_sentences += self.sentences[client]
            self._current_labels += self.labels[client]
        self.dataset_size = len(self._current_sentences)
        self.reset()

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
        sent_batch = [self._current_sentences[idx] for idx in idxs]
        sent_lens = np.array([len(sent) for sent in sent_batch])
        max_sent_len = np.max(sent_lens)

        # Encode sentences as glove vectors.
        sent_embeds = np.zeros((max_sent_len, bs, GLOVE_DIM))
        for i in range(len(idxs)):
            for j in range(len(sent_batch[i])):
                sent_embeds[j, i] = self.word_vec[sent_batch[i][j]]
        sent_embeds = torch.from_numpy(sent_embeds).float()

        # Convert labels to tensor.
        labels = torch.tensor([int(self._current_labels[idx]) for idx in idxs]).long()

        return sent_embeds, sent_lens, labels

    def has_next(self):
        """ Return True until all batches have been seen. """
        return self._next_pos < self.dataset_size

    def state(self):
        """ State of the data loader. """
        state = {
            "clients": self._clients,
            "current_sentences": self._current_sentences,
            "current_labels": self._current_labels,
            "permutation": self._permutation.copy(),
            "next_pos": self._next_pos,
            "step": self._step,
        }
        return state

    def load_state(self, state):
        """ Load a state for the data loader. """
        self._clients = state["clients"]
        self._current_sentences = state["current_sentences"]
        self._current_labels = state["current_labels"]
        self._permutation = state["permutation"]
        self._next_pos = state["next_pos"]
        self._step = state["step"]

    @property
    def n_words(self):
        return len(self.word_vec)

    @property
    def num_batches(self):
        return ceil(self.dataset_size / self.batch_size)
