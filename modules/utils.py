import torch
import numpy


def categorical(probs, num_samples=1, replacement=True):
    """Sample indices of probs with specified probabilities."""
    probs_norm = probs.flatten() / probs.sum()
    samples = torch.multinomial(probs_norm, num_samples, replacement).tolist()
    return numpy.asarray([numpy.unravel_index(x, probs.shape) for x in samples])
