import os
import sys
import torch

import numpy as np
from imageio import imwrite
from torchvision import datasets, transforms
from modules.utils import categorical

data = datasets.MNIST('./MNIST', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]))

def sequential_MNIST(num_samples, sequence_length, save=True, path="./dataset"):
    """Create sequences of digits from MNIST dataset."""
    data_path = path + "/data_" + str(sequence_length) + "_" + str(num_samples) + ".npy"
    label_path = path + "/labels_" + str(sequence_length) + "_" + str(num_samples) + ".npy"
    if save and os.path.exists(data_path) and os.path.exists(label_path):
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

def save_as_images(data, labels, path="./images"):
    """Save MNIST sequences as images."""
    if not os.path.exists(path):
        os.makedirs(path)
    for sample, label in zip(data, labels):
        img = np.concatenate(sample, axis=1)
        name = path + '/img_' + ''.join(map(lambda x: str(int(x)), label)) + '.png'
        imwrite(name, img.clip(0, 255).astype('uint8'))
        sys.stdout.write('Saved image: ' + name + '\n')

def create_ngram(sentences, n=5, c=10, save=True, path="./dataset"):
    """Create n-gram from set of sentences containing numbers from 0 to c-1."""
    ngram_path = path + "/" + str(n) + "gram_" + str(sentences.shape[1]) + "_" + \
        str(sentences.shape[0]) + ".npy"
    if save and os.path.exists(ngram_path):
        return np.load(ngram_path)
    ngram = np.zeros((c,) * n)
    for sentence in sentences.astype('uint8'):
        for i in range(len(sentence) - n + 1):
            ngram[tuple(sentence[i:i+n])] += 1
    if save:
        np.save(ngram_path, ngram)
        sys.stdout.write("Saved ngram file: " + ngram_path + "\n")
    return ngram

def sequential_MNIST_from_ngram(ngram, num_samples, replace=True):
    """Create sequences of digits from MNIST dataset according to given ngram probabilities."""
    dataset_data_split = {}
    c = ngram.shape[0]
    n = len(ngram.shape)
    for i in range(c):
        dataset_data_split[i] = data.train_data.numpy()[data.train_labels.numpy() == i]
    dataset = np.zeros((num_samples, n, 28, 28))
    labels = categorical(torch.tensor(ngram), num_samples)
    print(labels)
    for i in range(num_samples):
        for j in range(c):
            if (labels[i, ...] == j).any():
                pos = np.random.randint(len(dataset_data_split[j]), size=(labels[i, ...] == j).sum())
                dataset[i, (labels[i, ...] == j).nonzero()[0], ...] = \
                    dataset_data_split[j][pos]
    return dataset, labels