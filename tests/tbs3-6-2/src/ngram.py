import numpy
from collections import defaultdict


class Ngram(defaultdict):
    """Class implementation of n-gram probabilities in form of dictionary"""

    def __init__(self, n):
        """Create empty n-gram"""
        super(Ngram, self).__init__(int)
        self.n = n
        self.subgrams = dict()

    def sum(self):
        return sum([self[x] for x in self])

    def norm(self):
        """Normalize n-gram - required to call before sampling"""
        _s = self.sum()
        for _x in self:
            self[_x] /= _s
        return self

    def sample(self, x=None):
        """Sample random entry with corresponding probabilities."""
        if x is None:
            x = numpy.random.rand()
        for idx in self:
            x -= self[idx]
            if x < 0:
                return idx
        return None

    def parameters(self):
        for idx in self:
            yield self[idx]

    def __str__(self):
        obj = ''
        for idx in self:
            obj = '{}{}: {:.2f}%\n'.format(obj, idx, 100. * self[idx])
        return obj.rstrip()

    def size(self):
        return len(self)

    def subgram(self, v, cache=True):
        """Create subgram (unigram) for given entry.

        Parameters:

        - v (tuple of ints): prefix of length n-1 of some ngram entries (eg. '(1, 2)' for 3-gram)
        - cache (bool, optional): if set to `true`, then returns cached last result.

        Returns:
        - ngram: possible continuations of given prefix in form of unigram.
        """

        if len(v) != self.n - 1:
            raise NotImplementedError("""ngram.subgram does not handle indices resulting with k-grams for k > 1""")

        if cache and v in self.subgrams.keys():
            return self.subgrams[v]

        self.subgrams[v] = Ngram(1)

        for idx in self:
            ok = True
            for i in range(len(v)):
                if v[i] != idx[i]:
                    ok = False
            if ok:
                self.subgrams[v][idx[len(v)]] = self[idx]

        return self.subgrams[v].norm()
