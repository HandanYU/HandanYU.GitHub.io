---
layout: post
title: Discourse
summary: Learn about Dependency Grammar
featured-img: deep learning
language: chinese
category: NLP
---

![image1](/assets/img/post_img/121.png)

# 1. Discourse

**Discourse**: understanding how sentences relate to each other in a document

## 1.1 Discourse解决的三个关键任务

- **语篇分段（Discourse segmentation）**
  - 我们希望将document分割成多个独立的chunks
  - ![image1](/assets/img/post_img/122.png)
- **语篇解析（Discourse parsing）**
  - 试图将文档组织为一种 **层级结构（hierarchical structure）**
  - ![image1](/assets/img/post_img/123.png)
- **指代消解（Anaphora resolution）**
  - 消除文档中的指代词的歧义问题
  - ![image1](/assets/img/post_img/124.png)

# 2. 语篇分段（Discourse segmentation）

- 一个**document**可以被看作一个a sequence of **segments**

- **a segment**: 一段**连贯（cohesive）**的文字

- **连贯性 （cohesion）**: 连贯性意味着这段文字是围绕某个特定 **主题（topic）**或 **功能（function）**来组织的。

## 2.1 无监督方法 —— TextTiling algorithm

**Core idea**: 寻找句子之间具有**较低词汇连贯性 (low lexical cohesion)**的点

- 对于间隙(sentence gap) $$i$$
  - 创建两个BOW vectors，它们分别由间隙$$i$$上下$$k$$个句子中的单词组成
  - 计算两个向量之间的余弦相似度$$sim_i$$
  - 计算深度分数(**depth score**)：对于间隙$$i$$，$$depth(gap_i)=(sim_{i-1}- sim_i) + (sim_{i+1} - sim_i)$$
    - 对于第一个间隙，由于其前面没有其他间隙，所以在计算深度分数时我们可以直接忽略前项
  - 当depth score > **threshold** ($$t$$)，就在这个间隙处插入一个分界线。
- e.g., 这里我们将相关参数设为 $$k=1,t=0.9$$（即词袋向量来自间隙前后的各一个句子，深度分数的阈值为 0.9）：

[image1](/assets/img/post_img/125.png)

## 2.2 有监督方法

可以从*科学出版物*，*维基百科的文章*中通过以段落为边界进行分段，对段落间隙进行手动打正负标签。

- 如果前后两个段落之间涉及到章节之间的跳转，那么我们将这两个段落之间的间隙给予一个正标签，即我们将对这两个段落进行切分；
- 如果前后段落不涉及章节跳转，我们将给予段落间隙一个负标签，即我们会不对这两个段落进行切分。

### 2.2.1 有监督语篇分段器（Supervised Discourse Segmenter）

- binary classifier：用来识别边界，我们可以采用一个基于正负标签数据的二分类器来决定是否需要对给定的两个段落进行切分。
- sequence classifier：我们也可以使用像 HMM 或者 RNN 这类序列模型进行分类。这种情况下，我们在分段时会考虑一些上下文信息，从而在分段时得到一个全局最优的决策结果。

- Multi-class classifier：基于第一种，我们还可以对一个段落对应的不同章节进行分类。

# 3. 语篇解析（Discourse parsing）

- 目标是确定**语篇单元 (discourse units)** ，以及它们之间的关系，例如：某段文本是否是对另一段文本的解释。

- **修辞结构理论 (Rhetorical Structure Theory, RST)** 是一个对文档中的语篇结构进行层级分析的框架。运用于总结 (Summarisation)、问答 (QA) 等
- **解析（Parsing）**，即给定一段文本，我们希望 **自动化** 地构建出包含了语篇单元及其关系的 RST 树形结构。

## 3.1 **修辞结构理论 (Rhetorical Structure Theory, RST)** 

### 3.1.1 语篇单元 (discourse units)

- 通常是组成一个句子的 **子句（clauses）**
- 不跨越句子boundary

### 3.1.2 语篇单元之间的关系 Discouse Relations

- 连接 (conjuction)
- 论证 (justify)
- 让步 (concession)
- 详述 (elaboration) 等。

### 3.1.3 核心 (Nucleus) vs. 伴随体(Satellite)

- 在每个RST的Discouse Relations中，都有一个**主要论点**，即 **核心（nucleus）**。同时**支持论点**称为**伴随体（satellite）**
  - [image1](/assets/img/post_img/126.png)
- 但有些Discouse Relations是对等的（例如：连接 conjunction），这种情况下，两个论点都是**核心（nuclei）**。
  - [image1](/assets/img/post_img/127.png)

- 在RST中
  - 一定存在一个核心
  - 可能存在多个核心，或一个核心一个伴随体
  - 但不可能只有伴随体。

### 3.1.4 RST Tree

一个 RST 树的是通过迭代地对discourse units进行合并的方式构建的。将两个或者更多的语篇单元 (DU) 合并为一个复合语篇单元 (composite DU)

##### Bottom-Up Parsing

- Transition-based parsing
  - Greedy, uses shift-reduce algorithm
- CYK/chart parsing algorithm
  - Global, but some constraints prevent CYK from finding globally optimal tree for discourse parsing

##### Top-Down Parsing

- Segment document into DUs
- Decide a boundary to split into 2 segments
- For each segment, repeat step 2

## 3.2 Parsing Using Discourse Markers

- 一些语篇标记 (discourse markers)（提示语）显式地表明了关系。

  - 例如：although, but, for example, in other words, so, because, in conclusion,…

    这些单词表明了前后语篇单元之间存在某种新的关系。

- 我们可以利用这些语篇标记来构建一个简单的 **基于规则的解析器（rule-based parser）**。

  - 例如，我们可以收集所有这些显式的语篇标记，并试图理解其中包含的前后语篇单元之间的关系，然后构建一个简单的基于规则的解析器以用于语篇解析任务。

- 然而：

  - 许多关系根本没有用话语标记进行标记。

    可能存在这样的情况：某些语篇单元之间并没有显式的语篇标记，但是这些语篇单元之间确实存在某种关系。

  - 许多重要的话语标记（例如：and）具有歧义性（ambiguous）。

    - 有时并非用于话语标记
      例如：“John and I went fishing”，这里的 “and” 连接的只是句子中的两个主语单词，而非两个语篇单元。
    - 可能表示多种关系 同一个语篇标记单词可能表示多种关系，例如，“and” 有时可以表示连接关系（conjunction），有时则表示原因（reason）或者论证（justify）关系。因此，其中存在歧义性问题，我们不能总是基于简单规则将其视为连接关系。

## 3.3 Parsing Using Machine Learning

#### 3.3.2 RST Discourse Treebank 

- 300+ documents annotated with RST trees 

#### 3.3.1 Basic idea 

- Segment document into DUs 
- Combine adjacent DUs into composite DUs iteratively to create the full RST tree (bottom-up parsing)

## 3.4 Discourse Parsing Features

- Bag of words 
- Discourse markers 
- Starting/ending *n-*grams 
- Location in the text 
- Syntax features 
- Lexical and distributional similarities

## 3.5 Discourse Parsing Applications

- summarisation, 
- QA
- Argumentation, 
- Authorship attribution, 
- essay scoring.

# 4. Anaphora Resolution

### Features

- Binary features for number/gender compatibility 
- Position of antecedent in text 
- Include features about type of antecedent

