---
layout: post
title: Hidden Markov Model (HMM)
summary: 本篇文章主要讲解了HMM在NLP中sequence labelling的应用
featured-img: deep learning
language: chinese
category: NLP
---

![image-42](/assets/img/post_img/59.png)


# Formal Language Theory

为我们提供了一种定义语言的框架，它是一种数学框架。主要目的是为了解决 **从属问题 (membership problem)**：一个字符串是否属于某种语言。

- A language = set of strings
- A string = sequence of elements from a finite alphabet (vocabulary)

## 1. Regular language

- 任何 **正则表达式（regular expression）**都是一种正则语言。
  - 用于描述什么样的字符串是该语言的一部分

### 1.1 数学表示形式

- 从字母集中抽样得到的符号：$$\Sigma$$

- 空字符串：$$\epsilon$$
- 两个正则表达式的连接：RS
- 两个正则表达式的交替：R$$\lvert$$S
- 星号表示出现 0 次或者重复多次：R$$^*$$
- 圆括号定义运算的有效范围：()

### 1.2 性质

#### 1.2.1 封闭的（Closure)

对于**连接（concatenation）, 求并（union）, 求交（intersection）, 求反（negation）**这些操作是封闭的

This closure property enables us to generate a regular language by performing operations on regular languages. This allows us to factor NLP problems into small simple parts so that we can develop regular languages for each part and combine them into a complex system to handle the NLP problems.

## 2. Finite State Acceptor (FSA)

regular expression定义了regular language，但它不能给出一个algorithm来check一个string是否属于这个language。

- FSA给出了algorithm来check一个string是否属于regular language。

### 2.1 数学表示形式

- 输入字母集 $$\Sigma$$
- states: $$Q$$
- start state $$q_0\in Q$$
- final state $$F\subseteq Q$$
- transition function 

如果存在一条从 $$q_0$$到最终状态的 **路径（path）**，并且转移函数与路径上的每个符号都匹配，则接受该字符串。Djisktra 最短路径算法，O(Vlog⁡V+E)

### 2.2 Example

- Input alphabet : {a, b}

- States: {$$q_0, q_1$$}

- Start, final states, $$q_0$$, {$$q_1$$}

- Transition function : {($$q_0, a)\rightarrow q_0, (q_0,  b)\rightarrow q_1, (q_1, b)\rightarrow q_1$$}

  ![image-2](/assets/img/post_img/60.png)


  我们说接受a\*bb\*

### 2.3 FSA for Morphology (单词形态学)

我们回顾一下派生形态，是用来使用词缀（affixes）将一个单词变为另外一种语法类别的方法。（*表示无效派生）

![image-2](/assets/img/post_img/61.png)


下面就用FSA来表示一下


![image-2](/assets/img/post_img/62.png)

我们可以将上图进行一些压缩处理，合并一些具有共同路径的分支：


![image-2](/assets/img/post_img/63.png)

## 3. Weighted Finite State Acceptor (WFSA)

FSA 相当于一个二分类器，即某个单词是否可以添加相应的词缀来生成新的合法单词。但有时候我们需要有更多的记分性质，不是单纯的非黑即白的二分类问题。

例：fishful和disgracelyful，虽然两者都不是合法的英文单词，但fishful相较于disgracelyful 更可能成为合法的英文单词。musicky 和 writey，musicky更有可能是合法的形容词，

### 3.1 数学表示形式

- 状态集合：Q

- 输入符号的字母集：$$\Sigma$$
- 起始状态加权函数：$$\lambda: Q\rightarrow R$$
- 最终状态加权函数：$$\rho : Q\rightarrow R$$
- 转移函数：$$\delta: (Q,\Sigma, Q)\rightarrow R$$

### 3.2 WFSA 最短路径

对于路径 $$\pi = t_1,\dots, t_N$$，现在其总分为：

$$\lambda (t_0) + \sum_{i=1}^N \delta (t_i) + \rho(t_N)$$


### 3.3 N-gram 语言模型作为 WFSA

我们先回忆一下N-gram model，计算一个句子得分是用以下方法的

$$P(w_1, \dots, w_M) = \prob_{m=1}^M P_n(w_m\lvert w_{m-1},\dots, w_{m-n+1})$$

接下来运用WFSA表示

- state: $$q_0$$
- Transition score: $$\delta: (q_0,w, q_0)\rightarrow P_1(w)$$
- 假设初始状态和最终状态得分都是0

- 则序列得分为: $$\sum_{m=1}^M\delta(q_0, w_m,q_0)=\sum_{m=1}^MP_1(w_m)$$

## 4. Finite State Transducer (FST)

很多时候，我们并不希望只是接受字符串或者对其进行记分，我们还希望可以将其转换为另一种语言，并且进行相应的语法修正，并从句法上分析其结构等等。

对于FSA: 它可以将allure+ ing = allureing，但allureing并不是一个合法的单词，然而FSA无法处理这种情况，因为它只负责加词缀，不替换任何东西。于是FST就是来解决这个问题的。

### 4.1 数学表示形式

- 输入/出状态集合：Q
- 输入/出符号的字母集：$$\Sigma$$
- 起始状态加权函数：$$\lambda: Q\rightarrow R$$
- 最终状态加权函数：$$\rho : Q\rightarrow R$$
- 转移函数：$$\delta: (Q,\Sigma,\Sigma, Q)\rightarrow R$$
  - 其中第一个Q为输入状态，第一个$$\Sigma$$为输入字母集合；第二个Q为输出状态，第二个$$\Sigma$$为输出字母集合

### 4.2 **Edit Distance Automata**

![image-264](/assets/img/post_img/64.png)


- 若输入输出相同，则代价为0
- 若替换一个字母，删除一个字母，添加一个字母，代价都为1

则对于输入字母集合为aa，输出字母集合为ab，代价为1（因为替换了一个字母）

则对于输入字母集合为ab，输出字母集合为aaab，代价为2（因为添加了两个字母）

### 4.3 FST for Inflectional Morphology（屈折形态）

## 5. 自然语言

### 5.1 Regular language Vs Non-regular language

- 像下面这种句子，句子是无界的（递归的）。但是结构是有限局部的，这种称为**regular language**。如下面这个可以表示为(Det, Noun, Prep, Verb)*

![image-2](/assets/img/post_img/65.png)


- 对于能够表示为$$a^nb^n$$, 如“算术表达式（前后括号数量一致）”属于**non-regular language**，不能用正则表达式来表示

### 5.2 **Center Embedding**

![image-2](/assets/img/post_img/66.png)


假如我们有 2 个名词 cat 和 dog，那么其后也必须跟 2 个动词 chased 和 loves。这和之前$$a^nb^n$$的情况非常类似，因此这种情况下，我们的语言不再是正则的。因此，我们需要（至少是）**上下文无关语法 (context-free grammar)** 来描述这种情况，和正则语言相比，它具有更少的约束，可能捕获更多的自然语言的特性

