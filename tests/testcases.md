# List of testcases to perform

## MNIST dataset

### 1-grams

- [t1] small output (<= 3)
    - [t1-1] two letters, one 90%, second 10% (NOTE: weird behaviour - usually does not learn, but sometimes for an epoch it acheves good results???)
    - [t1-2] two letters, one 60%, second 40% (yields better results than [t1-1])
    - [t1-3] two letters, one 95%, second 5%
- [t02] large output (> 5)

### 3-grams with small output (out_dim < 5)

If out_dim = 5, then there are 125 possible combinations of 3-element sequences.

- [t3] sparse 3-gram (<=5 entries), out_dim = 4 or 5
    - [t3-1] out_dim = 4, ngram {(0, 1, 2): 90%, (1, 2, 3): 10%}
    - [t3-2] out_dim = 4, ngram {(0, 1, 2): 60%, (1, 2, 3): 40%} (NOTE: better learning curve than t3-1)
    - [t3-3] out_dim = 4, ngram {(0, 1, 2): 95%, (1, 2, 3): 5%}
    - [t3-4] out_dim = 5, ngram {(0, 1, 2): 80%, (1, 2, 3): 10%, (2, 3, 4): 10%}
    - [t3-5] out_dim = 5, ngram {(0, 1, 2): 20%, (1, 2, 3): 20%, (2, 3, 4): 20%, (3, 4, 0): 20%, (4, 0, 1): 20%} (NOTE: cyclic test)
    - [t3-6] out_dim = 5, ngram {(0, 1, 2): 50%, (1, 2, 3): 20%, (2, 3, 4): 30%}
- [t4] dense 3-gram (>20 entries)
    - [t4-1] randomized, 20 entries
    - [t4-2] randomized, 40 entries
    - [t4-3] randomized, 60 entries
    - [t4-4] sequence-like, 20 entries
    - [t4-5] sequence-like, 40 entries
    - [t4-6] sequence-like, 60 entries
