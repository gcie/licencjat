import os
import sys
import numpy as np
from imageio import imwrite
import matplotlib.pyplot as plt


def save_as_images(data, labels, path="./images"):
    """Save MNIST sequences as images."""
    if not os.path.exists(path):
        os.makedirs(path)
    for sample, label in zip(data, labels):
        img = np.concatenate(sample, axis=1)
        name = path + '/img_' + ''.join(map(lambda x: str(int(x)), label)) + '.png'
        imwrite(name, img.clip(0, 255).astype('uint8'))
        sys.stdout.write('Saved image: ' + name + '\n')


def plot_history(history):
    plt.plot(range(len(history['err_rate'])), history['err_rate'])
