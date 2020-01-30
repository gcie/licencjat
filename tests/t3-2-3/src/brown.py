import re
from nltk.corpus import brown
from src.ngram import Ngram


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
