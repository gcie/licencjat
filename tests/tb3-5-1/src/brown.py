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
    def __init__(self, n=3, dim=6, start=0, size=0, train=True):
        text = ''.join(brown.words()).lower()
        pattern = re.compile('[^' + 'etaoinsrhl'[:dim] + ']+')

        reduced = pattern.sub('', text)
        max_size = len(reduced)
        if max_size < start:
            raise f'Brown dataset reduced to {dim} most frequently occuring letters has length {max_size},\
                while you requested indices starting from {start}.'
        if max_size < start+size:
            warnings.warn(f'Size is too large. Brown dataset reduced to {dim} most frequently occuring letters has length \
                {max_size}, while you requested indices {start}:{start+size}. Clamping indices to {start}:{max_size}.')
            size = max_size - start
        targets = reduced[start:start+size]
        data = datasets.MNIST(MNIST_LOC, train=train, download=True)
        split_data = dict()
        for i in range(10):
            split_data[i] = data.data.numpy()[data.targets.numpy() == i]
        self.data = np.zeros((size, 28, 28), dtype='float32')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
        ctoi = {'e': 0, 't': 1, 'a': 2, 'o': 3, 'i': 4, 'n': 5, 's': 6, 'r': 7, 'h': 8, 'l': 9}
        for letter in ctoi:
            pass
