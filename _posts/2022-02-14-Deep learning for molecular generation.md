---
layout: post
title: Deep learning for molecular generation读书笔记
summary: 
featured-img: 梯度下降
language: english 
category: others
---

# 名词解释

## QED——药物相似性的定量估计

quantitative estimate of drug-likeness

**是一种将药物相似性量化为介于0和1之间的数值的方法**

## SAS —— 合成难度分数

 Synthetic Accessibility Score

a measurement of synthetic accessibility

## RO5

## latent vector



# 数据集

## ZINC

## ChemDiv

## SMILES



# The structure of the article

## introduce Common generative architecture

### RNN

### autoencoders

#### VAE (variational autoencoders)

1. use KL regularization
2. the posterior distribution in VAE is usually a Gaussian distribution with mean and variance predicted by the encoder

#### ChemVAE (Chemical VAE) 

- Mechanism

  1. encoder

     convert the discrete representations of molecules (SMILES strings in this case) into real-valued fix-dimensional continuous vectors (latent space).    

     - to ensure the validation of the decoded SMILES strings. —— add Gaussian noise with penalty term.

  2. decoder

     transform the vectors to SMILES strings

  3. predictor
     a predictive model based on multilayer perceptron to predict the molecular properties from latent space

- Improvement

  - Adopt the RDKit to filter the invalid ones
  - use **GP** (Gaussian Process) optimization method

- Objective

  $$5\times QED - SAS$$

####  GrammarVAE

utilized a context-free grammar (CFG) to form a parse tree

- Mechanism

  - Encoding

    - Parse SMILES string into a parse tree and decomposed the tree into a series of production rules

    - translating the production rules to one-hot vectors and every dimension of the vectors maps to a production rule

    - map the collection of one-hot vectors as a matrix to a latent vector

  - Decoding

    - parse a continuous vector to produce unnormalized log probablity vectors F and every dimension corresponds to a production rule
    - write the collection of F and rows of F could be used to select a sequence of valid production rules

  - Predicting

    

- Improvement

  -  teach VAE
  - provide smoother interpolation and more valid molecules
  - Use **BO** optimization method

- Objective of penalized logP

  $$\log P(m)-SAS(m)-ring\_penalty(m)$$

  

#### SD-VAE (syntax-directed VAE)

take both the syntax and semantics of SMILES into account



- Improvement
  - utilize the **CFG-based decoder** to capture the syntax of SMILES
  - as for encoder, introduce the two attributes (inherited and synthesized attributes) to the nonterminal symbols
  - Use **BO** optimization method
  - has smooth differences between neighboring molecules
  - has more complicated decoded structures

#### AAE (adversarial autoencoders)

a supervised method

1. use AL(adversarial learning) regularization -  match the posterior distribution to a prior distribution
2. the posterior distribution in AAE is encouraged to match a prior arbitrary distribution

#### ECAAE (entangled conditional AAE)

it is a semisupervised method

- Properties
  -  logP, SAS and E

- Improvement
  - integrates predictive and joint disentanglement approaches to solve disentanglement issues and inconsistency conditional generation problem
  - improve the interpretability of latent space.

### GANs (generative adversial networks)

- generative model
  learns a map from a prior to the data distribution to sample new data points
- discriminative model
  learns to classify whether samples come from the real data distribution rather than from G

#### ORGAN ( objective reinforced GAN )

the GAN architecture is combined with **reward functions with RL**

Cons

-  less effificient in controlling valid SMILES strings

#### ORGANIC

Cons

-  less effificient in controlling valid SMILES strings

#### RANC (Reinforced adversarial neural computer)

- Structure
  - generator (differentiable neural computer [DNC])
  - discriminaroe
  - objective functions
- Improvement
  -  DNC could afford the generation on long SMILES strings
- Cons
  -  less effificient in controlling valid SMILES strings

#### ATNC (Adversarial threshold neural computer)

Extension of RANC with a specific adversarial threshold block.

- based on **RL** approach

## creating molecules with desired properties based on SMILES

# optimization strategies

## TL 

- learn general properties on both large and small datasets

## BO

- global optimization of black-box functions

## GP (Gaussian process)

- model any smooth continuous function with few parameters
- only well suitable for the expressive AE-based models with a smooth latent space
- perform better than using random Gaussian search or the genetic algorithm

## RL

- joint training strategy of molecular generation with some reward objectives
- to reduce the risk of forgetting what was initially learned by the RNN
- to solve the dynamic decision problems

## CG

- based on conditional generative models
- easily applied to multiobjective optimization

### ReLeaSE

- combines with two deep neural networks
  - generative model: a stack-augmented RNN to learn hidden rules of forming
    sequences of letters for generating valid SMILES molecules

###  VAE-based de novo molecular generative model


##  summarize these generative models and make comparisons between them.