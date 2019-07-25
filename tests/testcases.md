# List of testcases to perform

## MNIST dataset

### 1-grams

- [t1] small output (<= 3)
    - [t1-1] two letters, one 90%, second 10%
    - [t1-2] two letters, one 60%, second 40%
    - [t1-3] two letters, one 95%, second 5%
- [t02] large output (> 5)

### 3-grams with small output (out_dim < 5)

If out_dim = 5, then there are 125 possible combinations of 3-element sequences.

- sparse 3-gram (<5 entries)
