---
layout: post
title: Lexical semantics
summary: 本篇文章主要讲解了HMM在NLP中sequence labelling的应用
featured-img: deep learning
language: chinese
category: NLP
---



# Word Semantics

## 1. Lexical semantics 

### 1.1 Classify words according to meaning aspect

#### 1.1.1 Meaning Through Dictionary

- 一词多义 polysemous: a word has multiple senses

#### 1.1.2 Meaning Through Relations

- 同义词 Synonymy & 反义词 Antonymy
- 上下位关系词 Hypernymy & 部分与整体关系词 Meronymy
  - Hyponyms: 下位词
  - Hypernyms：上位词


### 1.2. Lexical Database

#### 1.2.1 WordNet

- WordNet中的nodes包含的是senses of words而不是words也不是lemmas
- node通过synonyms集合的形式表示，也称为语法集（Synsets）
- e.g., "Bass"的Synsets: 
  -  {bass$$^1$$ , deep$$^6$$} 
  -  {bass$$^6$$ , bass voice$$^1$$ , basso$$^2$$} 

##### 1.2.1.1 Via nltk

通过Python中nltk调用

- 查看'bank'的所有Synset

```python
>>> nltk.corpus.wordnet.synsets('bank')
[Synset('bank.n.01'), Synset('depository_financial_institution.n.01'), Synset('bank.n.03'), 
Synset('bank.n.04'), Synset('bank.n.05'), Synset('bank.n.06'), Synset('bank.n.07'), 
Synset('savings_bank.n.02'), Synset('bank.n.09'), Synset('bank.n.10'), Synset('bank.v.01'), 
Synset('bank.v.02'), Synset('bank.v.03'), Synset('bank.v.04'), Synset('bank.v.05'), Synset('deposit.v.02'), 
Synset('bank.v.07'), Synset('trust.v.01')]  
```

- 查看

```python
>>> nltk.corpus.wordnet.synsets(‘bank’)[0].definition() 
u'sloping land (especially the slope beside a body of water)‘ 
```

- s 

```python
>>> nltk.corpus.wordnet.synsets(‘bank’)[1].lemma_names() 
[u'depository_financial_institution', u'bank', u'banking_concern', u'banking_company']
```

##### 1.2.1.2 Noun Relations in WordNet

<img src="/Users/yuhandan/Library/Application Support/typora-user-images/image-20220329212756220.png" alt="image-20220329212756220" style="zoom:50%;" />

### 1.3 Word Similarity

- 不像同义词（Synonymy）只是binary relation，**word similarity**是spectrum的。
- 我们可以使用lexical database或者同义词词典(thesaurus)去估算word similarity

#### 1.3.1 Word Similarity based on Path length

可以通过计算words在wordnet中的path length来估计它们之间的similarity

##### the shortest length between two senses

edgelen$$(c_1,c_2)$$

##### length between two senses

pathlen$$(c_1,c_2)=1+$$edgelen$$(c_1,c_2)$$

##### Similarity between two senses(synsets)

simpath$$(c_1, c_2)=\frac{1}{pathlen(c_1, c_2)}$$

##### Similarity between two words

wordsim$$(w_1, w_2)=\max_{c_1\in senses(w_1), c_2\in senses(w_2)}simpath(c_1, c_2)$$

- 相当于这两个words对应的synsets的最大距离

e.g. 

<img src="/Users/yuhandan/Library/Application Support/typora-user-images/image-20220329214357940.png" alt="image-20220329214357940" style="zoom:50%;" />

#### 1.3.2 Word Similarity based on Depth

由于我们可以通过刚上面的例子发现，money和richter scale两者与nickel的相似度差不多，但实际上我们知道money会更接近于nickel一点，造成这种现象的原因是因为money和richter scale它们的层数基本相同且距离它们公共的parent都很近，因此得到的edge差别不大。于是我们引入**depth information**。

**Lowest common subsumer （LCS）**

simwup$$(c_1, c_2)=\frac{2\times depth(LCS(c_1, c2))}{depth(c_1)+depth(c_2)}$$

- 其中$$LCS(c_1, c_2)$$表示sense $$c_1$$与sense $$c_2$$的最小公母(parent)

- e.g. simwup(*nickel,money*) = 2 * 2 / (6+3) =0.44

#### 1.3.3 Word Similarity based on Information Content

- aim to detect how abstract or generic the node is

- **Concept probability of a node**
  - general node: high concept probability （也就是对于更上面级别的word，更具有概括性，同时其拥有更多的children, e.g., objects）
  - narrow node: low concept probability
  - Sum up这个node的所有child的unigram probabilities
  - $$P(c)=\frac{\sum_{s\in child(c)}count(s)}{N}$$
  - 其中*child(c)*表示c的子语法集(synsets that are children of c)， N表示所有结点数。
  - e.g.
  - <img src="/Users/yuhandan/Library/Application Support/typora-user-images/image-20220330095600831.png" alt="image-20220330095600831" style="zoom:33%;" />
  - 对于synsets **geological-formation**，它的child(geological-formation) = {natural elevation, cave, shore, hill, ridge, grotto, coast}
  - 对于synsets **natural elevation**，它的child(natural elevation) = {hill, ridge}

- **Information Content**
  - $$IC=-\log P(c)$$，由于$$P(c)$$是很小的值，因此为了能够看出明显的差别，可以采用log-scale，同时为了求解相似度需要取negative
  - general concept: small IC
  - narrow concept: large IC
- **Similarity with IC**
  - simlin$$(c_1,c_2)=\frac{2\times IC(LCS(c_1,c_2))}{IC(c_1)+IC(c_2)}$$
  - **High** simlin: concept of **parent** is **narrow**, concept of **senses** are **general**

### 1.4 Task: Word Sense Disambiguation

#### 1.4.1 Concept

选择correct sense for words在一个sentence中

#### 1.4.2 Application

#### 1.4.3 Supervised WSD

- 通过标准的ML classifiers

- Cons:

  - context is ambiguous

  - the window size of the context ?
  - 需要sense-tagged corpora, time-consuming

#### 1.4.4 Unsupervised WSD

##### Lesk

通过对比context与该单词的wordnet gloss，选择overlap的单词数多的synset/sense

e.g.

<img src="/Users/yuhandan/Library/Application Support/typora-user-images/image-20220330103002009.png" alt="image-20220330103002009" style="zoom:33%;" />

这里我们可以看到context中有*deposits, mortgage*两个单词与*bank$$^1$$*这个synset重合而与*bank$$^2$$*没有重合单词，因此选择*bank$$^1$$*为bank在这个context中的sense

### 1.5 Lexical Database Problems

- 由于建立lexical database是manually的，造成
  - Expensive，耗费人力财力
  - Biased & noisy，由于human annotation
- 由于Language is dynamic，不断会有新词新意加入



