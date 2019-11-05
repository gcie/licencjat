# Wierd behaviour explanation

On the charts we can see weird behaviours:
- (a) on test model is learning zeroes better, while on data its exactly opposite
- (b) learning basically stops after few hundred epochs
- (c) sometimes it crashes and:
  1) in test, it starts learning zeroes for a brief moment quite well
  2) in data, it starts learning both features quite well

What happens? We have to take a look at our loss function (simplified for our case).

$$ \begin{aligned} 
\mathcal{L}(\theta, V) &= \frac{1}{B}\left(\sum_{x \in \mathcal{B}} p_{LM}(0)\cdot v_0\cdot p_{\theta}(y=0 | x) + \sum_{x \in \mathcal{B}} p_{LM}(1)\cdot v_1\cdot p_{\theta}(y=1 | x)\right) + p_{LM}(0)\cdot\ln(-v_0) + p_{LM}(1)\cdot\ln(-v_1) = \\
&= p_{LM}(0) \cdot \left(\frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=0 | x) \cdot v_0 + \ln(-v_0) \right) + p_{LM}(1) \cdot \left(\frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=1 | x) \cdot v_1 + \ln(-v_1) \right) = \\
&= p_{LM}(0)\cdot v_0 \cdot \frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=0 | x) + p_{LM}(1)\cdot v_1 \cdot \frac{1}{B}\sum_{x\in \mathcal{B}}p_{\theta}(y=1 | x)
\end{aligned} $$

If $v_0, v_1$ values are simila