import numpy


def categorical(probs, num_samples=1):
    """Sample indices of probs with specified probabilities."""
    return numpy.asarray([probs.sample(numpy.random.rand()) for _ in range(num_samples)])
