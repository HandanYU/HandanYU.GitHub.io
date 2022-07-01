---
layout: post
title: Contextual Representation
summary: Learn about contextual Representation, mainly focous on ELMo and BERT
featured-img: deep learning
language: chinese
category: NLP
---

![image1](/assets/img/post_img/103.png)

# Contextual Representation

我们之前学习到的word vectors/word embeddings，每个word都只有一种表达形式，也就是无论上下文是什么，只要是同一个word，它的word vector/word embedding都只有一种。不能捕捉到**单词的多义性** (multiple senses of words)。

而Contextual Representation，是一种基于上下文的单词表示。如果一个单词在两个句子中的含义不同，那么我们将得到该单词的两种不同的上下文表示。

# 1. RNN language model

 RNN 语言模型得到的单词的上下文表示仅仅捕获了该单词左边的上下文。

# 2. Bidirectional RNN

我们可以得到同时包含当前单词两侧信息的上下文表示

# 3. ELMo

**基于语言模型的嵌入（Embeddings from Language Models）**

- ELMo 在一个包含 1B（10 亿）单词的语料库上训练了一个**双向多层 LSTM 语言模型**。
- Combine **hidden states from multiple layers** of LSTM 用于 downstream tasks
  - 这是 ELMo 的创新点之一：因为之前关于预训练模型的上下文表示研究只使用了顶层的信息，因此并没有在性能上获得太大提升。而对于 ELMo，假如我们使用了一个 2 层的 LSTM，那么我们将同时使用第一层和第二层的 LSTM 的输出。
  - **lower layer representation = capture syntax**
    - 一般对于lower layer的contextual representation(e.g., LSTM第一层的隐藏状态) 倾向于捕获更多关于该单词的句法信息(syntax)。因此，非常适用于 **词性标注 (POS tagging)** 、**命名实体识别 (NER)** 等任务。
  - **higher layer representation = capture semantics**
    - 对于higher layer的contextual representation(e.g., 第二层 LSTM 中的隐藏状态)能捕获到的更多是关于单词语义方面的信息(semantics)，因此，更适用于一些理解相关任务，例如：**问答系统 (QA)** 、**文本蕴含 (textual entailment)** 、**情感分析 (sentiment analysis)** 等等。

## 3.1 ELMo Structure

- LSTM 层数 =2

- LSTM 隐藏层维度 =4096

- 使用 **字符级的卷积神经网络（Character CNN）**来创建Word Embedding

  - 例如：对于单词 “Playing”，相比直接创建一个该单词的词嵌入，ELMo 选择将其 token 化为一个个英文字母：“P”、“l”、“a”、“y”、“i”、“n”、“g”。然后我们学习得到单词中每个字母的字符嵌入，并且在其前后添加 paddings 以保证最终得到的单词嵌入的长度一致。然后将其喂给一个带最大池化层的 CNN 模型，来创建一个基于字符嵌入的单词 “Playing” 的表示。
  - 使用character based-token的原因是来解决unseen word的问题

  ![image1](/assets/img/post_img/104.png)

## 3.2 提取Contextual Representation

在得到训练好后的biLSTM，如何从中提取相应word的contextual representation呢？

简单的说就是：从两个方向的语言模型中提取所有层的隐藏状态，并对其进行连接，然后加权求和即可。

e.g., 希望得到“Let's stick to”中stick的contextual representation

- **Step1**:

​	将句子中的单词分别喂给forward LM，backward LM

![image1](/assets/img/post_img/105.png)

- **Step2**:

  将两个模型中对应的隐藏层状态以及输入层分别拼接在一起

- **Step3**：

  对每一层得到的连接向量进行加权求和

![image1](/assets/img/post_img/106.png)

## 3.3 ELMo运用与Downstream Task

只需要将该word的ELMo embedding与Downstream Task中该word的word embedding进行拼接。

![image1](/assets/img/post_img/107.png)

例如：这里，我们的下游任务中是一个简单的 RNN 模型，我们需要通过stick的上下文，得到它的POS。我们只需要在原始的隐藏状态 $$s_i$$ 的计算公式的基础上：将当前单词的嵌入 $$W_xx_i$$（其中，xi 为当前输入单词的 one-hot 向量，$$W_x$$ 为嵌入矩阵），连接一个基于 ELMo 得到的当前单词的嵌入 $$e_i$$ 即可。然后我们将得到的隐藏状态 $$s_i$$ 喂给 RNN 模型，然后像正常 RNN 模型一样进行训练即可。

## 3.4 ELMo的Limitation

由于ELMo本质是基于RNN的，因此仍存在一些缺点

- 依赖于**序列处理（Sequential processing）**：难以扩展到非常大的语料库和模型上。

  由于使用RNN，就会依赖于序列处理。也就是当我们需要得到一个单词的contextual representation的时候，我们必须知道它前面一个单词的contextual representation，同理需要得到前一个单词的contextual representation的时候，我们又必须知道前面的前面单词的contextual representation。因此我们必须从句子的第一个单词开始依次计算单词的contextual representation。因此对于这种方法，很难扩展到非常大的语料库和模型中去。

- **只能获取到表面的bidirectional representation information**

  对于bidirectional RNN，由于我们只是简单的将独立的forward LM, backward LM的每层的输出进行了拼接操作，其实它们之间不存在交互。

# 4. Disadvantage of Contextual embedding

- Difficult to do intrinsic evaluation (e.g. word similarity, analogy)
- 可解释性差
- Computationally expensive to train large-scale contextual embeddings

# 5. BERT

![image1](/assets/img/post_img/108.png)

-  **基于 Transformers 的双向编码器表示** （Bidirectional Encoder Representations from Transformers， BERT）

-  使用 **自注意力网络（self-attention networks）**，又称 **Transformers**，来捕获单词之间的依赖关系。
   - 优点是：无须序列处理
-  BERT 还使用 **掩码语言模型（masked language model）**来捕获deep bidirectional representations
-  无法进行生成操作，也就是BERT 无法单独进行语言生成任务，它无法计算有效的句子概率，也无法从左至右生成句子

## 5.1 掩码语言模型（masked language model）

Concept: 随机mask $$k\%$$的words，然后希望能正确预测出这些被masked的words

## 5.2  Next Sentence Prediction

**目标**：预测B是否是A的下一个句子

**不需要带标签的语料库**:我们同样可以通过类似negative sampling的方法，自己从语料库中创建出isNextSentence以及notNextSentence样本

- 对于正样本，我们可以选择语料库中前后两个句子组成一个pair作为一个instance
- 对于负样本，我们同样可以从语料库中先选择一个句子，然后在选择一个不是它下一句句子组成一个pair作为一个instance.

## 5.3 How to use BERT

- 用pre-trained BERT，在downstream task上进行train (i.e., fine-tue)
- 只需要在BERT的structure上加一层classifier layer

### 5.3.1 Spam Detection

![image1](/assets/img/post_img/109.png)

- Input BERT: append a special token (i.e., [CSL]) at the start of every sentence
- through BERT, obtain the contextual representation of each word
- Input Classifier: only input the contextual representation of the special token (i.e., [CSL])

## 5.4 Attention

在BERT中使用**Attention**来替换RNNs (or CNNs)来捕捉words之间的dependencies.

### 5.4.1 Self-Attention

- **Input**: (in vector forms， 它们都是来自单词嵌入的线性投影（linear projections）)
  - query: the target word
  - key k and value v: the key and value of context word
- **Step1**
  - 通过比较query vector of target word与key vectors of context words，得到每个context word对应的weight
  - $$W_i=\frac{e^{qk_i}}{\sum_je^{qk_j}}$$
- **Step2**
  - 对于每个context word，将对应的weight与其value vector作乘积
  - $$W_iv_i$$
- **Step3**
  - 对所有加权后的context words vector求和，得到target word的contextual representation
  - **基于self-attention的单个单词的上下文表示**： $$A(q,K,V) = \sum_iW_iv_i$$
  - 对于multiple queries，也就是target word有很多的时候，可以将 stack them in a matrix (Q)
  - $$A(Q,K,V) = softmax(QK^T)V$$，因为我们进行了很多点积操作，所以最终得到的值可能非常大。为了防止值过大，使用**scaled dot-product** 
    - $$A(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$， 其中$$d_k$$是query vector的dimension

![image1](/assets/img/post_img/110.png)

#### Example

假设现在我们有sentence: ` I made her duck`。我们试图得到`made`的contextual representation。我们已知`made`的embedding通过linear projections得到的一个向量$$q$$，`her`的embedding通过两次不同的linear projections得到的两个向量分别记为$$k_{her}, v_{her}$$。同理对于其他几个context words（i.e., I, duck)，也经过两次linear projections分别得到对应的两个向量。于是根据计算公式我们可以得到`made`的attention值。其中红色部分由softmax方法计算得到。

![image1](/assets/img/post_img/111.png)

### 5.4.2 Multi-Head Attention

![image1](/assets/img/post_img/112.png)

- 之前对于each word pair (one target word - one context word)只有一个attention
- Multi-head Attention是通过上述方法进行多次（可以是并行），得到多个attention，并将它们进行拼接
- **Step1**: 对每个$$head_i$$运用不同的线性投影矩阵$$W_i$$进行转换得到$$QW_i^Q, KW_i^K, VW_i^V$$.
- **Step2**: 对于每个head，计算一个Attention。同样的Q, K, V对于不同head，投影矩阵$$W$$不同
  - $$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$
- **Step3**：经过scaled dot-product
- **Step4**: 将所有head的Attention拼接起来 $$Concat(head_1, \dots, head_h)$$
- **Step5**: 再进行一次linear-projection最终得到MultiHead Attention
  - $$MultiHead(Q,K,V) = Concat(head_1, \dots, head_h)W^O$$

## 5.5 Transformers

BERT中主要由一个个**Transformer Block**组成

![image1](/assets/img/post_img/113.png)

### Input Embedding

![image1](/assets/img/post_img/114.png)



- 其中Position Embedding十分重要，因为在做Attention的时候，我们只是简单的进行word pair组合，完全没有考虑位置关系，因此通过Position Embedding 可以引入位置关系信息。

# ELMo Vs. BERT

- 我们可以看到对于ELMo，两个模型在中间过程是没有任何交集的，只有在输出的时候进行了合并。且单独对于每个模型，它们都只有一个方向。

![image1](/assets/img/post_img/115.png)

- 而对于BERT，其中每个蓝色椭圆表示 Transformer。我们可以发现在中间过程中，每个Transformer都会先汇集所有contextual words的信息，然后一起transform。因此它们能够同时学到来自不同方向上下文的信息。同时由于最后一层的输出于最终结果之间是一一对应的，因此可以方便不断增加深度。

![image1](/assets/img/post_img/116.png)

