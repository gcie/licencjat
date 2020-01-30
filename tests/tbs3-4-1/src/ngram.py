import warnings
import numpy as np
from collections import defaultdict


class Ngram(defaultdict):
    """Class implementation of n-gram probabilities in form of dictionary"""

    def __init__(self, n):
        super(Ngram, self).__init__(int)
        self.n = n
        self.subgrams = dict()

    def sum(self):
        return np.sum([self[x] for x in self])

    def norm(self):
        _s = self.sum()
        for _x in self:
            self[_x] /= _s
        return self

    def sample(self, x):
        for idx in self:
            x -= self[idx]
            if x < 0:
                return idx
        return None

    def parameters(self):
        for idx in self:
            yield self[idx]

    def show(self):
        for idx in self:
            print(idx, ' -> {:.5f}%'.format(100. * self[idx]))

    def get(self, i):
        for idx in self:
            if not i:
                return self[idx]
            i -= 1
        raise AttributeError

    def ravel(self):
        warnings.warn("Ngram ravel is deprecated, use get(0) instead")
        for idx in self:
            return self[idx]

    def size(self):
        return len(self)

    def subgram(self, v):
        if v in self.subgrams.keys():
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
                