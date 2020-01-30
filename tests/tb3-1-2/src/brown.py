import re
from nltk.corpus import brown
from src.ngram import Ngram


def strtotuple(vowels):
    ctoi = {'a': 0, 'e': 1, 'y': 2, 'u': 3, 'i': 4, 'o': 5}
    return tuple([ctoi[x] for x in vowels])


def get_brown_ngram(n=3):
    text = ''.join(brown.words()).lower()
    pattern = re.compile('[^aeyuioAEYUIO]+')
    vowels = pattern.sub('', text)
    ngram = Ngram(n)
    for i in range(len(vowels) - n + 1):
        ngram[strtotuple(vowels[i:i+n])] += 1
    return ngram.norm()
