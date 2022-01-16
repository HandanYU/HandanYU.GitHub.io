---
layout: post
title: 损失函数 Loss Function
summary: 本章主要总结了机器学习中典型的损失函数类型及其特征，以及对应的典型机器学习模型。
featured-img: 损失函数

---

##### Table of Contents  
- [0-1 Loss](#0-1 Loss)  
- [Binary cross entropy / log Loss](#Binary cross entropy / log Loss) 
- [Square Loss](#Square Loss)  
- [Exponential Loss](#Exponential Loss)  
- [Hinge Loss / Max Margin Loss](#Hinge Loss / Max Margin Loss)  
- [Perceptron Loss](#Perceptron Loss)  
- [Cross-entropy Loss](#Cross-entropy Loss)
- [Maximum Likelihood Estimation](#Maximum Likelihood Estimation)

<a name="0-1 Loss"/>

# 0-1 Loss

- used in Preceptron
- $$L(Y,\hat{Y})$$ = 1, if \|$$Y-\hat{Y}$$\| $$\geq T.$$ else 0
- non-convex function

<a name="Binary cross entropy / log Loss"/>

# Binary cross entropy / log Loss

- used in LR
- L(Y,P(Y\|X)) = -$$\log$$P(Y\|X)
- L = -$$\frac{1}{N}\sum_{i=1}^Ny_i\log \hat{y_i}+(1-y_1)\log (1-\hat{y_i})$$
    - where $$\hat{y_i}=p(y_i$$\|$$X)$$
- sensitive to noise compared with [Hinge Loss / Max Margin Loss](#Hinge Loss / Max Margin Loss)
- describe distribution of feature probability well

<a name="Square Loss"/>

# Square Loss

- used in Regression,
- L = $$\sum_{N}(y_i-\hat{y_i})^2$$

<a name="Exponential Loss"/>

# Exponential Loss

- used in AdaBoost
- sensitive to outliers and noises
- L=$$\frac{1}{N}\sum_N\exp(-y_i\hat{y_i})$$

<a name="Hinge Loss / Max Margin Loss"/>

# Hinge Loss / Max Margin Loss

- L=$$\max (0, 1-y_i\hat{y_i})$$
    - if classify correctly, return 0; else $$1-y_i\hat{y_i}$$
- used in SVM

<a name="Perceptron Loss"/>

# Perceptron Loss

- L = $$\sum_N\max (0,-\hat{y_i})$$
- advanced hinge loss

<a name="Cross-entropy Loss"/>

# Cross-entropy Loss

- L = $$-\frac{1}{N}\sum_Ny_i\log \hat{y_i}$$
    - where $$\hat{y_i} = \frac{\exp(z_i)}{\sum_K\exp(z_i)}$$
    - $$z_i$$ is the output of class $$i$$


<a name="Maximum Likelihood Estimation"/>

# Maximum Likelihood Estimation

- write down Loss Function
- derivation Loss Function
- let equal to 0
- compute optimum parameters’ values