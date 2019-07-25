import os
import sys
import warnings
import pickle
from time import gmtime, strftime

import torch
import numpy as np
from torchvision import datasets, transforms
from modules.ngram import Ngram
from config import BATCH_SIZE


def categorical(probs, num_samples=1):
    """Sample indices of probs with specified probabilities."""
    return np.asarray([probs.sample(np.random.rand()) for _ in range(num_samples)])


def dump_data_to_local(content, fname=None):
    if fname is None:
        fname = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.pkl'
    output = open(fname, 'wb+')
    pickle.dump(content, output)
    output.close()
    return fname


def read_data_from_local(fname):
    return pickle.load(open(fname, "rb"))


class SequentialMNIST(torch.utils.data.Dataset):
    """Create dataset containing sequences of MNIST digits based on given ngram probabilities"""
    def __init__(self, ngram, num_samples, sequence_length=None):
        if sequence_length is not None:
            warnings.warn("Variable sequence_length is unused. Sequence generated have \
                length equal to n (from ngram)")
        data = datasets.MNIST('./MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
        split_data = dict()
        for i in range(10):
            split_data[i] = data.train_data.numpy()[data.train_labels.numpy() == i]
        self.data = np.zeros((num_samples, ngram.n, 28, 28))
        self.targets = categorical(ngram, num_samples).astype('int64')
        for i in range(num_samples):
            for j in range(10):
                if (self.targets[i, ...] == j).any():
                    pos = np.random.randint(len(split_data[j]), size=(self.targets[i, ...] == j).sum())
                    self.data[i, (self.targets[i, ...] == j).nonzero()[0], ...] = \
                        split_data[j][pos]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def sequence_loader_MNIST(ngram, num_samples):
    data = SequentialMNIST(ngram, num_samples)
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def train_loader_MNIST():
    data = datasets.MNIST('./MNIST', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def test_loader_MNIST():
    data = datasets.MNIST('./MNIST', train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)


def sequential_MNIST(num_samples, sequence_length, load=True, save=True, path="./dataset"):
    """Create sequences of digits from MNIST dataset."""
    data = datasets.MNIST('./MNIST', train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    data_path = path + "/data_" + str(sequence_length) + "_" + str(num_samples) + ".npy"
    label_path = path + "/labels_" + str(sequence_length) + "_" + str(num_samples) + ".npy"
    if load and os.path.exists(data_path) and os.path.exists(label_path):
        return np.load(data_path), np.load(label_path)
    dataset_data = np.zeros((num_samples, sequence_length, 28, 28))
    dataset_labels = np.zeros((num_samples, sequence_length))
    for i in range(num_samples):
        sys.stdout.write("\rGenerating sample " + str(i+1) + "...")
        p = np.random.choice(60000, sequence_length)
        dataset_data[i, ...] = data.train_data.numpy()[p]
        dataset_labels[i, :] = data.train_labels.numpy()[p]
    sys.stdout.write('\rDone!\n')
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(data_path, dataset_data)
        sys.stdout.write("Saved data file: " + data_path + "\n")
        np.save(label_path, dataset_labels)
        sys.stdout.write("Saved labels file: " + label_path + "\n")
    return dataset_data, dataset_labels


def create_ngram(sentences, n):
    """Create n-gram dictionary from set of sentences."""
    ngram = Ngram(n)
    for sentence in sentences.astype('int64'):
        for i in range(len(sentence) - n + 1):
            ngram[tuple(sentence[i:i+n])] += 1
    return ngram.norm()
