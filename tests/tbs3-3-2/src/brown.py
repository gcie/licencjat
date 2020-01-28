import re
from nltk.corpus import brown
from src.ngram import Ngram
import torch
from config import BATCH_SIZE, MNIST_LOC
from torchvision import datasets, transforms
import numpy as np
import warnings


def strtotuple(vowels):
    ctoi = {'e': 0, 't': 1, 'a': 2, 'o': 3, 'i': 4, 'n': 5, 's': 6, 'r': 7, 'h': 8, 'l': 9}
    return tuple([ctoi[x] for x in vowels])


def get_brown_ngram(n=3, dim=6):
    text = ''.join(brown.words()).lower()
    pattern = re.compile('[^' + 'etaoinsrhl'[:dim] + ']+')
    vowels = pattern.sub('', text)
    ngram = Ngram(n)
    for i in range(len(vowels) - n + 1):
        ngram[strtotuple(vowels[i:i+n])] += 1
    return ngram.norm()


class BrownMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, n, dim, start=0, size=0, train=True, log=True):
        self.n = n
        text = ''.join(brown.words()).lower()
        pattern = re.compile('[^' + 'etaoinsrhl'[:dim] + ']+')
        reduced = pattern.sub('', text)
        max_idx = len(reduced)
        if size == 0:
            size = max_idx - start
        if max_idx < start:
            raise f'Brown dataset reduced to {dim} most frequently occuring letters has length {max_idx},\
                while you requested indices starting from {start}.'
        if max_idx < start+size:
            warnings.warn(f'Size is too large. Brown dataset reduced to {dim} most frequently occuring letters has length \
                {max_idx}, while you requested indices {start}:{start+size}. Clamping indices to {start}:{max_idx}.')
            size = max_idx - start
        self.size = size
        ctoi = {'e': 0, 't': 1, 'a': 2, 'o': 3, 'i': 4, 'n': 5, 's': 6, 'r': 7, 'h': 8, 'l': 9}
        self.targets = np.array([ctoi[c] for c in reduced[start:start+size]], dtype='int64')
        data = datasets.MNIST(MNIST_LOC, train=train, download=True)
        split_data = dict()
        for i in range(10):
            split_data[i] = data.data.numpy()[data.targets.numpy() == i]
        self.data = np.zeros((size, 28, 28), dtype='float32')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        used = dict()
        for i in range(dim):
            used[i] = (self.targets == i).sum()
            max_i = len(split_data[i])
            self.data[self.targets == i, ...] = np.array([split_data[i][j % max_i] for j in range(used[i])])
        if log:
            for i in used:
                print('Used {:.2f}% of available {:d}\'s ({:.2f} times each)'.format(min(1, used[i] / len(split_data[i])) * 100, i, used[i] / len(split_data[i])))
            print('Used {:.2f}% of Brown sequence'.format(100 * size / max_idx))

    def __len__(self):
        return self.size - self.n + 1

    def __getitem__(self, index):
        res = self.transform(self.data[index])
        for i in range(1, self.n):
            res = torch.cat((res, self.transform(self.data[index+i])), 0)
        return res, self.targets[index:index+self.n]


def sequential_BrownMNIST(n, dim, start=0, size=0, train=True):
    data = BrownMNISTDataset(n, dim, start, size, train)
    return torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
