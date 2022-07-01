---
layout: post
title: Dependency Grammar 
summary: Learn about Dependency Grammar
featured-img: deep learning
language: chinese
category: NLP
---

# Dependency Grammar

## 1. Pros

- Dependency grammar offers a simpler approach 

  - describe **relations** between pairs of words 

  - namely, between **heads** and **dependents**

- Deal better with languages that are morphologically rich and have a relatively free word order

- Head-dependent relations similar to semantic relations between words 

## 2. Dependency Relations

Captures the grammatical relation between: 

- **Head** = central word 

- **Dependent** = supporting word

Grammatical relation = subject, direct object, etc 

![image1](/assets/img/post_img/117.png)

## 3. Properties of a Dependency Tree

- Each word has a single head (parent) 
- There is a single root node 
- There is a unique path to each word from the root 
- All arcs should be **projective**

## 4. Projectivity

- An arc is projective if there is a path from head to every word that lies between the head and the dependent

- Dependency tree is projective if all arcs are projective

- a dependency tree is projective if it can be drawn with no **crossing edges**
- Dependency trees generated from constituency trees are **always projective**

![image1](/assets/img/post_img/118.png)

## 5. Dependency Parsing

Find the best structure for a given input sentence ，寻找出a series of actions

### 5.1 Transition-based Parsing

e.g., bottom-up greedy method 

can only handle **projective** dependency trees!

Less applicable for languages where crossdependencies are common

- root -> 动词
- 动词 -> 宾语/介词宾语（e.g., in a boat的boat)/宾语从句中的动词
- 动词 -> 主语
- 动词 -> and, or
- 动词 -> 情态动词
- 宾语 -> 介词
- 名词 -> 冠词
- 名词 -> 形容词
- 名词 -> 连词

![image1](/assets/img/post_img/119.png)

- action可以从[SHIFT, RIGHTARC, LEFTARC]中选择，
  - 如果stack中最后的两个之间没有relation，则选择SHIFT （也就是将buffer中的第一个word添加到stack中
  - 如果stack中最后的两个之间有relation，则根据dependencies，选择RIGHTARC / LEFTARC
    - 同时需要进行一步SHIFT: 将箭头指向的后者从stack中删除 
  - 注意如果删除的那个元素是之后的起始word，则不应该选择这个action
- 直到stack中只剩下root

#### 5.1.1  oracle

convert dependency grammer tree into a series of actions

Given a dependency tree, the role of oracle is to generate a sequence of ground truth actions

![image1](/assets/img/post_img/120.png)

#### 5.1.2 Parsing Model

We then train a supervised model to **mimic** the actions of the **oracle**

##### Parsing As Classification

- Input: 
  - Stack (top-2 elements: *s**1* and *s**2*) 
  - Buffer (first element: *b**1*) 

- Output 
  - 3 classes: *shift*, *left-arc*, or, *right-arc* 

- Features 
  - word (*w*), part-of-speech (*t*)

##### Classifier

 SVM works best 

**Weakness**: local classifier based on greedy search 

**Solutions**: 

- Beam search: keep track of top-N best actions 
- Dynamic oracle: during training, use predicted actions occasionally 
- graph-based parser

### 5.2 **Graph-based Parsing**

Given an input sentence, construct a fully-connected, weighted, directed graph 

- **Vertices**: all words 
- **Edges**: head-dependent arcs 
- **Weight**: score based on training data (relation that is frequently observed receive a higher score) 
- **Objective**: find the maximum spanning tree (Kruskal’s algorithm)

#### Pros

- Can produce non-projective trees 
  - Not a big deal for English 
  - But important for many other languages 

- Score entire trees 
  - Global optimal
  - Avoid making greedy local decisions like transition-based parsers 
- Captures long dependencies better

#### Cons

Caveat: tree may contain cycles 

- Solution: need to do cleanup to remove cycles (Chu-Liu-Edmonds algorithm)