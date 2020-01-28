# Unsupervised Sequence Classification using Sequential Output Statistics

[Paper](https://arxiv.org/pdf/1702.07817.pdf)

# List of testcases

## 1-grams

- [x] **t1** small output (<= 3)
    - [x] **t1-1**: two letters, one 90%, second 10% (NOTE: weird behaviour - usually does not learn, but sometimes for an epoch it achieves okay-ish results???)
    - [x] **t1-2**: two letters, one 60%, second 40% (yields better results than **t1-1**)
    - [x] **t1-3**: two letters, one 95%, second 5%
- [ ] **t2** large output (> 5)

## 3-grams with small output (out_dim <= 5)

If out_dim = 5, then there are 125 possible combinations of 3-element sequences.

- [ ] **t3**: sparse 3-gram (<=5 entries), out_dim = 4 or 5
    - [x] **t3-1**: out_dim = 4, ngram {(0, 1, 2): 90%, (1, 2, 3): 10%} (PRZEROBIĆ t3-1-5!!! loss on test, overfitting)
    - [x] **t3-2**: out_dim = 4, ngram {(0, 1, 2): 60%, (1, 2, 3): 40%} (NOTE: better learning curve than t3-1)
    - [ ] **t3-3**: out_dim = 4, ngram {(0, 1, 2): 95%, (1, 2, 3): 5%}
    - [x] **t3-4**: out_dim = 5, ngram {(0, 1, 2): 80%, (1, 2, 3): 10%, (2, 3, 4): 10%}
    - [x] **t3-5**: out_dim = 5, ngram {(0, 1, 2): 20%, (1, 2, 3): 20%, (2, 3, 4): 20%, (3, 4, 0): 20%, (4, 0, 1): 20%} (NOTE: cyclic test)
    - [x] **t3-6**: out_dim = 5, ngram {(0, 1, 2): 60%, (1, 2, 3): 10%, (2, 3, 4): 10%, (3, 4, 0): 10%, (4, 0, 1): 10%} (NOTE: imbalanced cyclic test)
    - [x] **t3-7**: out_dim = 5, ngram {(0, 1, 2): 50%, (1, 2, 3): 20%, (2, 3, 4): 30%}
- [ ] **t4**: dense 3-gram (>10 entries)
    - [ ] **t4-1**: randomized, 10 entries
    - [ ] **t4-2**: randomized, 20 entries
    - [ ] **t4-3**: randomized, 40 entries
    - [ ] **t4-4**: sequence-like, 20 entries
    - [ ] **t4-5**: sequence-like, 40 entries
    - [ ] **t4-6**: sequence-like, 60 entries

## 3-grams with large output

- TODO

## 5-grams

- TODO

## 7-grams (out dim = 10)

- [ ] **t7-1**: randomized, 3 entries
- [ ] **t7-2**: randomized, 5 entries, 0.02 minimum ngram variance

## 10-gram

- [ ] **t10-1**: 1 entry

## Brown n-gram based tests

- [ ] **tb3-1**: 2000 sequences of length 30, generated from brown 3-gram with mapping a->0, e->1, y->2, u->3, i->4, o->5. Model has access only to original n-gram.
- [ ] **tb3-2**: 2000 sequences of length 30, generated from brown 3-gram with mapping a->0, e->1, y->2, u->3, i->4, o->5. Model has access only to post-generation n-gram.
- [x] **tb3-3-1**: 2000 sequences of length 30 (n=3, dim=3, 'aey', post-ngram) generated from brown 3-gram with mapping a->0, e->1, y->2. Model has access only to post-generation n-gram.
- [x] **tb3-3-2**: 2000 sequences of length 30, generated from brown 3-gram with mapping a->0, e->1, y->2. Model has access only to post-generation n-gram. USE LARGER MINIBATCHES
- [ ] **tb3-4-1**: 2000 sequences of length 30 (n=3, dim=4, mbs=1024, 'etao', orig-ngram)
- [x] **tb3-4-2**: 2000 sequences of length 30 (n=3, dim=4, mbs=1024, 'etao', post-ngram) 
- [x] **tb3-5-1**: 5000 sequences of length 30 (n=3, dim=5, mbs=1024, 'etaoi', ???)
- [x] **tb4-1-1**: 2000 sequences of length 30 (n=4, dim=3, mbs=1024, 'etaoi', post-ngram)

## Brown sequence based tests

- [x] **tbs3-3-1**: sequence of length 100000 (n=3, dim=3, mbs=1024, 'eta', full ngram) (dzieją się dziwne wahania po 250 epokach)
- [x] **tbs3-3-2**: sequence of length 100000 (n=3, dim=3, mbs=1024, 'eta', post ngram) 
- [x] **tbs3-4-1**: sequence of length 100000 (n=3, dim=4, mbs=1024, 'etao', full ngram)
- [x] **tbs3-4-2**: sequence of length 100000 (n=3, dim=4, mbs=1024, 'etao', post ngram)
- [x] **tbs3-5-1**: sequence of length 100000 (n=3, dim=5, mbs=1024, 'etaoi', full ngram)
- [ ] **tbs3-5-2**: sequence of length 100000 (n=3, dim=5, mbs=1024, 'etaoi', post ngram)
- [x] **tbs3-6-1**: sequence of length 100000 (n=3, dim=6, mbs=1024, 'etaoin', full ngram)
- [x] **tbs3-6-2**: sequence of length 100000 (n=3, dim=6, mbs=1024, 'etaoin', post ngram)
- [ ] **tbs3-7-2**: sequence of length 100000 (n=3, dim=7, mbs=1024, 'etaoins', post ngram)
- [ ] **tbs3-8-2**: sequence of length 100000 (n=3, dim=8, mbs=1024, 'etaoinsr', post ngram)

# Conclusions

## 1-grams

On the charts we can see weird behaviours:
- (a) learning basically stops after few hundred epochs
- (b) sometimes it crashes and it starts learning both features quite well

What happens? We have to take a look at our loss function (simplified for our case):

$$ \begin{aligned} 
\mathcal{L}(\theta, V) &= \frac{1}{B}\left(\sum_{x \in \mathcal{B}} p_{LM}(0)\cdot v_0\cdot p_{\theta}(y=0 | x) + \sum_{x \in \mathcal{B}} p_{LM}(1)\cdot v_1\cdot p_{\theta}(y=1 | x)\right) + p_{LM}(0)\cdot\ln(-v_0) + p_{LM}(1)\cdot\ln(-v_1) = \\
&= p_{LM}(0) \cdot \left(\frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=0 | x) \cdot v_0 + \ln(-v_0) \right) + p_{LM}(1) \cdot \left(\frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=1 | x) \cdot v_1 + \ln(-v_1) \right) = \\
&= p_{LM}(0)\cdot v_0 \cdot \frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=0 | x) + p_{LM}(1)\cdot v_1 \cdot \frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=1 | x) + C
\end{aligned} $$

At start our model will learn to minimize this loss, hence while $p_{LM}(0)\cdot v_0 > p_{LM}(1)\cdot v_1$ holds (remember that $v_i < 0$), then it will be penalized for predicting zeroes and rewarded for predicting ones, resulting in constant model predicting only ones. And hence $v_i$ are sampled from uniform distribution it will happen $95\%$ of the time.


## 3-grams with small output

When distribution is balanced:
- learning curve is better (there are no letters that occur rarely)
- model is more prone to yielding wrong results - predicts letters wrongly (cyclic test)

Why?

Distribution match is randomized at start - it starts with values from interval [0, 1] and hence it might *swap* distributions of some items, especially for very sparse n-grams.


## Other notices:
- sometimes model will become so sure that it will predict some values and be almost 100% certain (distribution close to [0, 0, ..., 0, infity, 0, ... 0]) which causes numerical problems (but can be easily avoided - as we know that it is in its best form we can stop learning)
