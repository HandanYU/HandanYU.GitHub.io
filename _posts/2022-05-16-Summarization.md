---
layout: post
title: Summarisation
summary: Learn about Summarisation
featured-img: deep learning
language: chinese
category: NLP
---
# Summarisation

Distill the most important information from a text to produce shortened or abridged version

- **outlines** of a document 
- **abstracts** of a scientific article 
- **headlines** of a news article 
- **snippets** of search result

## 1. Goal

### 1.1 **Generic summarisation**

Summary gives important information in the document(s) 

### 1.2 **Query-focused summarisation**

- Summary responds to a user query 
- "Non-factoid" QA 
- Answer is much longer than factoid QA

## 2. **Extractive summarisation**

Use **top-ranked** (highest saliency or highest centrality or nucleus) sentences as extracted summary

### 2.1 as for single-doc

**Information ordering** & **Sentence realisation** are not necessary for single document extractive summarisation.

### Content selection

- select what sentences to extract from the document
- Find sentences that are important or **salient**
- **Pre-process**: remove unnecessary words (functional words, stop words)

#### 2.1.1 TF-IDF

$$
weight(w)=tf_{d,w}\times idf_{w}
$$

##### Saliency of A Sentence

$$
weight(s)=\frac{1}{\lvert S\lvert}\sum_{w\in S}weight(w)
$$

#### 2.1.2  Log Likelihood Ratio

a word is salient if its probability in the **input corpus** is very different to a **background corpus**
$$
weight(w) = 1,~~~~~~ -2\log\lambda(w)>10\\
weight(w)=0,~~~~~~~~~~~~~~~~~~~ otherwise
$$

##### Saliency of A Sentence

$$
weight(s)=\frac{1}{\lvert S\lvert}\sum_{w\in S}weight(w)
$$

#### 2.1.3 Sentence Centrality

Measure distance between sentences, and choose sentences that are closer to other sentences 

- sentences are represented in TF-IDF BOW.
- compute distance using cosine similarity

$$
centrality(s)=\frac{1}{\#sentence}\sum_{s'}cos_{tfidf}(s,s')
$$

#### 2.1.4 RST Parsing

- Nucleus more important than satellite 

- A sentence that functions as a nucleus to more sentences = more salient

### 2.2 as for multi-doc

#### Challenges

- Redundancy in terms of information 

- Sentence ordering

#### 2.2.1 Content Selection

- select salient sentences
  - same as single-doc, but **ignore sentences that are redundant**
- Maximum Marginal Relevance
  - Iteratively select the best sentence to add to summary 
  - Sentences to be added must be **novel**
  - Penalise a candidate sentence if it’s **similar** to extracted sentences: 
    - $$MMR-penalty(s)=\lambda\max_{s_i\in S}sim(s, s_i)$$
  - Stop when a desired number of sentences are added

#### 2.2.2 Information Ordering

- **Chronological ordering:** 
  - ​	Order by document dates 
- **Coherence:** 
  -  Order in a way that makes adjacent sentences similar 
  -  Order based on how entities are organised (centering theory)

#### 2.2.3 Sentence Realisation

Make sure entities are referred coherently

- Full name at first mention 
- Last name at subsequent mentions

Apply coreference methods to first extract names 

Write rules to clean up

## 3. **Abstractive summarisation** 

Paraphrase

### 3.1 Model

#### 3.1.1 Encoder-decoder

Input: Source sentence (document)

Output: Target sentence (summary)

#### 3.1.2 Encoder-decoder with Attention Mechanism

Improvement:

- Attention mechanism 
- Richer word features: POS tags, NER tags, tf-idf 
- Hierarchical encoders 
- One LSTM for words 
- Another LSTM for sentences

