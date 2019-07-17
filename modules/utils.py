import torch
import numpy as np


def categorical(probs, num_samples=1):
    """Sample indices of probs with specified probabilities."""
    return np.asarray([probs.sample(np.random.rand()) for _ in range(num_samples)])