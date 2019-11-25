### 1-grams

Interesting results, one can write whole chapter on why and how it does not work.

### 3-grams with small output

When distribution is balanced:
- learning curve is better (there are no letters that occur rarely)
- model is more prone to yielding wrong results - predicts letters wrongly (cyclic test)

Why?

Distribution match is randomized at start - it starts with values from interval [0, 1] and hence it might *swap* distributions of some items, especially for very sparse n-grams.


### Other notices:
- sometimes model will become so sure that it will predict some values and be almost 100% certain (distribution close to [0, 0, ..., 0, infity, 0, ... 0]) which causes numerical problems (but can be easily avoided - as we know that it is in its best form we can stop learning)
- 