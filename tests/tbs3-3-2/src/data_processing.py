import os
import pickle
import sys
from time import gmtime, strftime

import numpy as np
import torch
from torchvision import datasets, transforms

from config import BATCH_SIZE, MNIST_LOC
from src.ngram import Ngram


def categorical(probs, num_samples=1):
    """Sample indices of probs with specified probabilities."""
    return np.asarray([probs.norm().sample(np.random.rand()) for _ in range(num_samples)])


def dump_data_to_local(content, fname=None):
    if fname is None:
        fname = strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.pkl'
    output = open(fname, 'wb+')
    pickle.dump(content, output)
    output.close()
    return fname


def read_data_from_local(fname):
    return pickle.load(open(fname, "rb"))


def generate_sequences(ngram, num_samples, sequence_length, return_ngram=False):
    ngram.norm()
    targets = np.zeros((num_samples, sequence_length), dtype='int64')

    def gen_sequence(i):
        targets[i, :ngram.n] = np.array(ngram.sample(np.random.rand()))
        for j in range(1, sequence_length - 2):
            x = ngram.subgram(tuple(targets[i, j:j+2])).sample(np.random.rand())
            if x is None:
                return False
            targets[i, j+2] = np.array(x)
        return True
    for i in range(num_samples):
        while not gen_sequence(i):
            pass
    return targets


class SequentialMNIST(torch.utils.data.Dataset):
    """Create dataset containing sequences of MNIST digits based on given ngram probabilities"""
    def __init__(self, ngram, num_samples, sequence_length=None, train=True):
        if sequence_length is None:
            sequence_length = ngram.n
        data = datasets.MNIST(MNIST_LOC, train=train, download=True)
        split_data = dict()
        for i in range(10):
            split_data[i] = data.data.numpy()[data.targets.numpy() == i]
        self.data = np.zeros((num_samples, sequence_length, 28, 28), dtype='float32')
        self.n = ngram.n
        if ngram.n == sequence_length:
            self.targets = categorical(ngram, num_samples).astype('int64')
        else:
            self.targets = generate_sequences(ngram, num_samples, sequence_length)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
        for i in range(num_samples):
            if i % 97 == 0:
                sys.stdout.write("\rGenerating sample " + str(i+1) + "...")
            for j in range(10):
                if (self.targets[i, ...] == j).any():
                    pos = np.random.randint(len(split_data[j]), size=(self.targets[i, ...] == j).sum())
                    self.data[i, (self.targets[i, ...] == j).nonzero()[0], ...] = \
                        split_data[j][pos]
        sys.stdout.write('\rDone!                                                                        \n')

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.n + 1)

    def __getitem__(self, index):
        sample = index // (self.data.shape[1] - self.n + 1)
        offset = index % (self.data.shape[1] - self.n + 1)
        res = self.transform(self.data[sample][offset])
        for i in range(1, self.n):
            res = torch.cat((res, self.transform(self.data[sample][offset + i])), 0)
        return res, self.targets[sample][offset:offset+self.n]

    def get(self, index):
        res = self.transform(self.data[index][0])
        for i in range(1, self.data.shape[1]):
            res = torch.cat((res, self.transform(self.data[index][i])), 0)
        return res, self.targets[index]


def sequence_loader_MNIST(ngram, num_samples, sequence_length=None, train=True):
    data = SequentialMNIST(ngram, num_samples, sequence_length=sequence_length, train=train)
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def train_loader_MNIST():
    data = datasets.MNIST(MNIST_LOC, train=True, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)


def test_loader_MNIST():
    data = datasets.MNIST(MNIST_LOC, train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ]))
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)


def sequential_MNIST(num_samples, sequence_length, load=True, save=True, path="./dataset"):
    """Create sequences of digits from MNIST dataset."""
    data = datasets.MNIST(MNIST_LOC, train=True, download=True,
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


def randomized_ngram(n, size, out_dim=10, min_var=1e-6):
    """Create randomized n-gram"""
    ngram = Ngram(n)
    while ngram.size() < size:
        ngram[tuple(np.random.randint(0, out_dim, n))] = np.random.random()
    unique = set()
    for idx in ngram:
        for i in idx:
            unique.add(i)
    if len(unique) != out_dim:
        return randomized_ngram(n, size, out_dim, min_var)
    ngram.norm()
    mu = sum(ngram.values()) / size
    var = sum([(x - mu) ** 2 for x in ngram.values()]) / size
    if var < min_var:
        return randomized_ngram(n, size, out_dim, min_var)
    print(f"Ngram variance: {var}")
    return ngram


def sequence_ngram(n, entries, out_dim=10):
    """Create sequence-based n-gram"""
    ngram = Ngram(n)
    idx = np.random.randint(0, out_dim, n)
    while ngram.size() < entries:
        ngram[tuple(idx)] = np.random.random()
        idx = np.append(idx[1:], np.random.randint(0, out_dim))


def retrieve_ngram(sequence_loader, n):
    """Retrieve ngram from data loader"""
    ngram = Ngram(n)
    for _, y in sequence_loader:
        for sample in y:
            ngram[tuple(sample.to('cpu').numpy())] += 1
    return ngram.norm()
