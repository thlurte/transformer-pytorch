 <h2 align="center">Transformer Architecture</h2>
<p align="center">A network architecture based on attention mechanism.</p>
<br>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/The-Transformer-model-architecture.png/800px-The-Transformer-model-architecture.png" alt="Transformer Network Architecture">
</p>

### Input Embeddings
Here learned embeddings are used to convert the input tokens and output tokens to vectors of dimension $d_{model}$ and weights are multiplied by $\sqrt{d_{model}}$ .

### Positional Encoding
To provide some information about the relative or absolute position of the tokens in the sequence "positional encodings" are added to the input embeddings at the bottoms of the encoder and decoder stacks.

The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed.

> Denominator is calculated in log space.

Sine and cosine functions of different frequencies are used:

$$
PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}}) \\
$$

$$
PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d_{model}}) \\
$$

In the input vector sine function is applied to the even positions and cos is applied to the odd positions.

### Layer Normalization
This is where mean and variance are calculated independently for each batch and new value is calculated for each of them, Also **gamma** and **beta** are introduced to provide some fluctuation in the data. 

$$
\hat{x_{j}}=\frac{x_{j}-\mu_{j}}{\sqrt{\sigma^2_{j}+\epsilon}}
$$

### Feed Forward Network
Each of the layers in the encoder and decoder contains a fully connected feed-forward network. This consists of two linear transformations with a ReLU activation in between.

$$
FFN(x)=max(,xW_{1}+b_{1})W_{2}+b_{2}
$$

The dimensionality of input and output is $d_{model}=512$ and, the inner-layer has dimensionality $d_{jj}=2048$.

### Multi-head Attention
Multi-head attention takes input from the positional encoding and uses it three times. Where $Q$ means query, $K$ means key and $V$ means value. These three input matrices then are multiplied by weights and split into `h` matrices. (These matrices are split along embedding dimension).

$$
Attention(K,Q,V)=softmax\Big(\frac{QK^T}{\sqrt{d_{k}}}\Big)V
$$

$$
head_{i}=Attention(QW_{i}^Q,KW_{i}^K,VW_{i}^W)
$$

Then attention is applied to each of the split matrices and the result matrices are concatenated and multiplied by $W$ to get output.

$$
MultiHead(K,Q,V)=Concat(head_{i}...head_{n})W^{O}
$$

In this work $h=8$, parallel attention layers, or heads are employed. For each of these $d_{k}=d_{c}=d_{model}/h=64$.

### Encoder
The encoder is composed of a stack of $N = 6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. residual connection is employed around each of the two sub-layers, followed by layer normalization.


