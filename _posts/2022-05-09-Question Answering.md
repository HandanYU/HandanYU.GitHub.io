---
layout: post
title: Question Answering
summary: Learn about Question Answering
featured-img: deep learning
language: chinese
category: NLP
---

![image1](/assets/img/post_img/154.png)
# Question Answering

automatically determining the answer for a natural language question

## 1. Factoid & non-factoid Question

in NLP， we mainly focus on **Factoid Question**

-  They are easier 
-  They have an objective answer 
-  Current NLP technologies cannot handle non-factoid answers 
-  There’s less demand for systems to automatically answer non-factoid questions

### 1.1 Factoid Question

- have short precise answers
- What war involved the battle of Chapultepec? 
- What is the date of Boxing Day? 
- What are some fragrant white climbing roses? 
- What are tannins?

### 1.2 Non-factoid Questions

 require a longer answer, critical analysis, summary, calculation and more:

- Why is the date of Australia Day contentious?
- What is the angle 60 degrees in radians?

## 2. Approach

### 2.1 Information retrieval-based QA 

![image1](/assets/img/post_img/155.png)

- Given a query, search relevant documents
- Find answers within these relevant documents

#### 2.1.1 Question Processing

- Find key parts of question that will help retrieval
  - Discard non-content words/symbols (wh-word, ?, etc) 
  - Formulate as tf-idf query, using unigrams or bigrams 
  - Identify entities and prioritise match
- Reformulate question using templates
  - E.g. “Where is Federation Square located?” 
  - Query = “Federation Square located” 
  - Query = “Federation Square is located [in/at]”
- Predict expected answer type (here = LOCATION)

#### 2.1.2 Predict Answer Type

##### feature

Head word (中心词)

##### model

multi-classifier

#### 2.1.3 Retrival

find relevant document -> find really relevant passages(sentences/paragraphs)

##### features

- instances of question keywords
- name entities of the answer type
- proximity of the terms in the passage
- rank by IR engine

#### 2.1.4 Answer Extraction

find concise answer

##### tasks

- predict where the answer span starts and ends in the passage

##### models

- LSMT based
  - Issues: too complex
- BERT based
  - use self-attention can learn much more information from question and passage





### 2.2 Knowledge-based QA 

#### issues

convert natural language into triple is not trival

entity linking is ambiguous

- Builds semantic representation of the query 
- Query database of facts to find answers

### 2.3 Hybrid Methods

#### Core Idea of Watson

- Generate lots of candidate answers from text-based and knowledge-based sources 
- Use a rich variety of evidence to score them 
- Many components in the system, most trained separately