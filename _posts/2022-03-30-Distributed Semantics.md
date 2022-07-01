---
layout: post
title: Distributional Semantics/Representation
summary: Learn about Distributional Semantics/Representation
featured-img: deep learning
language: chinese
category: NLP
---

![image1](/assets/img/post_img/128.png)

![image1](/assets/img/post_img/129.png)


# Distributional Semantics/Representation

# 1. Distributional Hypothesis

- 你可以通过其周围的上下文单词来了解一个目标单词
- Document as context
- Word window as context

# 2. Produce Word Vector

## 2.1 Count-based method

>  通过不同的上下文信息所捕获到的关系是不一样的，如果我们采用 TF-IDF，我们捕获到的语义学信息会更加宽泛，通常和某种主题关联；如果我们采用 PMI/PPMI，我们捕获到的词向量更多是关于局部单词上下文的语义学信息。

### 2.1.1 Document as context

#### Vector Space Model (VSM)

![image1](/assets/img/post_img/130.png)


通过两个不同视角维度：

- 如果我们观察**每一行**，我们可以将其视为**每个文档的词袋模型**表示；
- 如果我们观察**每一列**，我们可以将其视为**每个单词的词向量**表示。

#### TF-IDF（**Term Frequency-Inverse Document Frequency**）

基于VSM，对矩阵中的值加权处理。

- 我们首先可以得到一个 **TF（term-frequency）矩阵**，和之前一样，单元格中的数字表示该单词在对应文档中出现的频率。

![image1](/assets/img/post_img/131.png)


- 然后，我们将计算每个单词对应的 **IDF（inverse document frequency）**值。

$$
idf_w=\log_2 \frac{|D|}{df_w}
$$

其中|D|为document的数量，$$df_w$$表示拥有$$w$$的document数量。e.g.,如上图，假设一共有500个document，求country的IDF值。|D| = 500, $$df_{country}=14$$，则$$idf_{country}=5.158$$.

- 接着对每个单元格分别求$$TF \times IDF$$，得到如下TF-IDF矩阵。e.g., 例如425-country对应单元格，$$5\times 5.158=26$$

![image1](/assets/img/post_img/132.png)


#### 降维 Dimensionality Reduction

我们可以发现对于出现频率特别高的单词对应的tf-idf值基本为0，且对于没有出现过的单词的tf-idf值也是0。因此tf-idf矩阵十分的稀疏。

- **intuitive**: 我们可以创建更短、更密集的向量。通过减少特征，消除噪声等方法。

##### 奇异值分解 （Singular Value Decomposition，SVD）

- 一种流行的降维方法。其核心思想就是将一个矩阵A，分解为三个矩阵相乘。也就是将A降维得到我们感兴趣的U.
- SVD与特征分解的区别：SVD不需要要求是方阵。
- 下图即为TF-IDF进行SVD的过程，其中矩阵U是word视角，$$V^T$$是document视角

![image1](/assets/img/post_img/133.png)


##### 截断：潜在语义分析（Latent Semantic Analysis, LSA）

> LSA 与 LDA 的区别：

​	在SVD的基础上可以更进一步对矩阵U，$$\Sigma$$, $$V^T$$进行截断。对于U矩阵，截取前k列作为最终的word vectors，相当于原先一个word的embedding dim = m，然后截取后得到embedding dim = k。因此k不能太大过于接近于m会不起到任何降维作用。【得到的U（$$V^T$$）的每列（每行）是根据特征值大小按顺序进行排列的。】

![image1](/assets/img/post_img/134.png)

### 2.1.2 Words as context

- 构建一个矩阵，其中每个值是对应word与其他words（word window中的其他words）一起出现的counts。因此矩阵的行和列都是word，每个单元格表示所在行所在列的单词在documents中共同出现的次数。

![image1](/assets/img/post_img/135.png)


- 存在problems: 同样使用的是raw frequency，因此会出现dominated by common words的问题

#### 点互信息 (Pointwise Mutual Information, PMI)

为解决majority domain的问题，我们引入PMI。
$$
PMI(x, y) = \log_2 \frac{P(x,y)}{P(x)P(y)}
$$
其中$$x, y$$分别表示两个word，$$P(x,y)$$表示$$x,y$$在documents中同时出现的概率，$$P(x), P(y)$$分别表示$$x,y$$在documents中出现的概率。

- 如图，$$P(state, country)=10/15871304，P(state) = 12786 / 15871304, P(country) = 3617 / 15871304$$

![image1](/assets/img/post_img/136.png)


- 按上述方法求解PMI，得到PMI矩阵

![image1](/assets/img/post_img/137.png)


- 从上图中我们可以发现PMI在semantic方面表现良好，如heaven和hell的PMI高达6.61，说明这两个单词极度相似

- 但仍存在一些**问题**:

  - Biased to rare word pairs，也就是对于低频率的词，它所对应的所有PMI都会相应偏大
    - 因为PMI的分母是独立先验概率的乘积，当P(x)很小的时候，P(x)P(y)就很小，相反得到的PMI会很大。
    - 为了避免这个问题，可以采用n-gram类似的smoothing方法
      - Normalized PMI = $$\frac{PMI(x,y)}{-\log P(x,y)}$$，此时$$P(x,y)$$很小，而$$-\log P(x,y)$$很大，从而可以起到综合分母过大的情况。
  - 对于unseen的word pair无法很好的处理，我们可以发现它对应的PMI=-inf
    - 为了避免这个问题，可以采用将所有负值取 0（i.e., Positive PMI, PPMI）

> 对于Words as context的count-based method, SVD同样适用于对PMI矩阵进行降维

## 2.2 Neural Methods

通过采用Neural Network Models来构造Word Embeddings

### 2.2.1 Word2Vec

#### Core Idea

- Embedding of **target words** 应该与Embedding of **neighbouring words  相似**， 同时与不出现在其附近的word的Embedding**不相似**

#### Core tasks

Word2Vec主要任务还是训练一个分类器$$f(x) \rightarrow y$$，（其中x是target word, y 是neighbouring word)。分类的任务是判断$$x, y$$两个words一起出现是否符合语义。

#### Skip Gram Model

![image1](/assets/img/post_img/138.png)


用一个词语作为输入，来预测它周围的上下文

##### 输入/输出

- 输入：one-hot (target word x的one-hot encoding vector)

  - > 如何表示x呢？由于model的输入必须是数值型，且还不能使用word2vec得到x的word vector，因此使用one-hot

- 输出：需要预测的context word 的一个 1×|V| 的概率向量，其中每个元素代表词汇表中相应位置的单词在这里出现的概率

  - 最后可以接一层softmax： $$P(w_{t-1}=V_i|V_j)=\frac{\exp(W_jC_i)}{\sum_{u\in V}\exp(W_jC_u)}$$其中$$V$$表示vocabulary中words经过排序后的列表。

- Loss: cross-entrpy

- 其实模型中我们的最重要的产出是隐藏层的权重，也就是我们最终需要生成的word embedding matrix【所有的隐藏层没有激活函数】

  - 如上图所示，其中$$W$$即是target word的word embedding matrix，对于第$$i$$行代表第$$i$$个word的target word embedding
    - 因为通过输入目标word的one-hot encoder，经过与W连乘，得到的就是$$W_i$$（W的第$$i$$行）
  - C是context word的word embedding matrix，对于第$$i$$行代表第$$i$$个word的context word embedding
  - 当我们需要对目标单词计算embedding时，我们使用矩阵 W；当我们需要对上下文单词计算embedding时，我们使用矩阵 C。

- 问题： 当我们在给定target word的情况下，计算一个context word的概率时，我们需要将词汇表中所有单词对应的word embedding的点积进行累加，这个过程非常缓慢

![image1](/assets/img/post_img/139.png)


- 通过将Skip gram model转化为binary classifier，即将原来计算**是context word的概率**任务简化为判断**是否为context word**

##### 负采样(Negative Sampling)

- 采用**负采样(Negative Sampling)**

  - 通过随机从vocabulary中选取部分word作为负样本（也就是non-context word)

  - > 为什么可以采用随机选取的方法？ 因为vocabulary的size很大，其中属于一个target word的context word数量相比而言很少，因此可以通过随机选择得到的word极大概率上属于non-context word。即使个别错选，也不会很大程度影响最终结果。

- 接着可以采用Logistic进行二分类

  - 对于**正样本**，我们希望target word和**context word**之间的相似度尽可能**高**. $$P(+|t, c) = \frac{1}{1+e^{-tc}}$$
  - 对于**负样本**，我们希望target word和**non-context word**之间的相似度尽可能**低**.$$P(-|t,c) = 1-\frac{1}{1+e^{-tc}}$$

- 损失函数

  - 对于一个target word/context word pair (t, $$c_{pos}$$)它的Loss
  - $$\log (P(+|t,c_{pos}))+\sum_{(t,c)\in -}\log(P(-|t,c))$$
  - 但一般情况，我们会选择k个negative sample，也就是
  - $$\log (P(+|t,c_{pos}))+\sum_{i=1}^k\log(P(-|t,n_i))$$，其中$$n$$是负样本集合

##### 特性

- 无监督：由于没有label，所有的正负样本是sample的
- 高效
  - 负采样（避免在整个词汇表上计算 softmax）
  - 可以扩展到非常大的语料库上
    还有一些在非常大的语料库上预训练的词向量，我们也可以直接下载使用。

# 3. Problems with word vectors/embeddings

- Difficult to quantify the quality of word vectors
- Don’t capture polysemous words

# 4. Evaluation

## 4.1 Word Similarity

- cosine similarity
- predition with human intuition

## 4.2 Word 类比(Analogy)

e.g., Man is a king, Women is a XX.

v("Man") - v("king") = v("Women") - v(XX)

v(XX) = v("Women") - v("Man") + v("king")

也就是我们预测出来XX的word vector要尽可能与v(XX)相近

## 4.3 Embedding Space

![image1](/assets/img/post_img/140.png)


通过将word embedding绘制在二维图中，我们可以发现"woman"-"man"的路径方向与"queen"-"king"的路径方向相近。"slow"-"slower"-"slowest"与"short"-"shorter"-"shortest"路径相近

## 4.4 Downstream Tasks

最佳的evaluation是通过将其运用与其他downstream task中。

