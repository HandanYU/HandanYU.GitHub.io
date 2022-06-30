---
layout: post
title: Molecular
summary: 本篇文章主要讲解了HMM在NLP中sequence labelling的应用
featured-img: deep learning
language: chinese
category: others
---
## Approaches
### Sequence-based
- take SMILES representation as the input
- supervised method: RNN
- unsupervised method: Mol2vec
- pretrained model: SMILES-BERT
### Graph-based
applied the convolutional layers to encode molecular graph into neural fingerprints
- basic model : GNN
- extensions: graph attention network (GAN), message passing neural network (MPNN)
