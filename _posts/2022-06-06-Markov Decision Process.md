---
layout: post
title: Markov Decision Processes
summary: 本章主要总结了面试中涉及到的机器学习相关的一些问题。
featured-img: machine learning
language: chinese 
category: AI
---

# Markov Decision Processes

classical planning假设了action are **deterministic**，然而*Markov Decision Processes (MDPs)*没有这个假设，而是认为每个action都有multiple outcomes，并且每个outcome又一个probability

## 1. Intuition

- 抛硬币
- 掷骰子

## 2. Definition

- a *fully observable, probabilistic* state model
- the most common formulation of MDPs is **Discounted-Reward Markov Decision Process**

### 2.1 Discounted-Reward Markov Decision Process

$$(S, s_0, A, P, r,\gamma)$$

- $$S$$: a state space
- $$s_0$$: initial state $$s_0\in S$$
- $$A$$: actions $$A(s) \subseteq A$$
- $$P$$: transition probabilities $$P_a(s'\lvert s)$$ for $$s\in S$$ and $$a\in A(s)$$
- $$r$$: rewards, $$r(s, a, s')$$ , positive or negative of transitioning from state $$s$$ to state $$s′$$ using action 
- $$\gamma$$: discount factor, $$0\leq \gamma < 1$$
  - 当$$\gamma$$=1的时候，我们不会对未来回报打折，这就退回到啦最基本的*additive reward*，我们不能取1，因为我们总是选择会对未来回报打折
  - 当$$\gamma$$接近于0的时候，我们认为未来回报无关紧要，只在乎眼前利益

#### 2.1.1 Discount factor

确定**future reward**与**current reward**相比应**discount**多少

> 我们为什么会对future reward进行discount呢？

作为humans，我们通常会奖励最近的，相比于未来的。比如你希望今天得到\$100，还是一年后得到\$100，显然我们会更希望今天就能得到。

假设经过一系列actions后，agent得到的rewards序列为$$r_1, r_2, r_3, \dots$$， 另$$\gmama$$是discount factor，则得到的**discount rewards**为 (给予future较少的reward，也就是对future reward进行discount)

$$ V=r_1+\gamma r_2+\gamma^2 r_3 + \gamma^3r_4+\dots\\=r_1+\gamma(r_2+\gamma(r_3+\gamma(r_4+\dots)))$$

则有递推公式$$V_t=r_t + \gamma V_{t+1}$$, 当前discount reward = 当前reward + future reward by discounted



### 2.2 Differences between Classical planning

- transition function is not deterministic
- there is no goal states
- there are no action costs, action costs are modelled as *negative rewards*
- have a *discount factor*

### 2.3 Example MDP: Grid World

## 3. Policies

Instead of a sequence of actions in classical planning, an MDP produces a **policy**

**policy** $$\pi$$ 用来告诉agent： *which is the best action to choose in each state*

### 3.1 Deterministic policies vs. Stochastic policies

#### 3.1.1 Deterministic policy

- $$\pi: S\rightarrow A$$ 
- Mapping states to actions, 明确了在state $$s$$应该选择的action
- $$\pi(s)$$: 在state $$s$$处应该执行的action
- store as *dictionary-like* object

#### 3.1.2 Stochastic policy

- $$\pi: S\times A\rightarrow \mathbb{R}$$
- $$\pi(s)$$，明确了在state $$s$$处，*probability distribution of actions can be chosen*
- $$\pi(s, a)$$： 在state $$s$$处，执行action $$a$$的probability
- 一般情况下，我们选择$$arg\max_a \pi(s, a)$$作为state $$s$$处应该执行的action

## 4. Optimal Solutions for MDPs

**aim**: maximise *expected discounted accumulated reward* from the **initial state $$s_0$$**

### 4.1 Expected discounted reward

从state $$s$$开始的expected discounted reward可以定义为如下：

$$V^{\pi}(s)=E_{\pi}[\sum_i\gamma^i r(s_i, a_i, s_{i+1})\lvert s_0=s, a_i=\pi(s_i)]$$

表示了根据policy $$\pi$$ , from state $$s$$的**expected value**

### 4.2  Bellman equation

Bellman equation 假设一旦我们知道best states，我们总能知道通向best state的action

- 首先我们已经知道对于state $$s$$， 它当前时刻$$t$$的discount reward为： $$r(s) + \gamma V_{t+1}$$
- 假设我们执行a从state $$s$$到state $$s'$$， 则有transition probability $$P_a(s'\lvert s)$$
- 于是我们可以将执行a从state $$s$$到state $$s'$$得到的expected discounted reward表示为: $$P_a(s'\lvert s)[r(s, a, s') + \gamma V_{t+1}]$$
- 因此我们将所有可到达的state得到的expected discounted reward累加得到$$\sum_{s'\in S}P_a(s'\lvert s)[r(s, a, s') + \gamma V_{t+1}]$$ ，其实这式子被称为**Q-value**
  - **Q-value**: $$Q(s, a)=\sum_{s'\in S}P_a(s'\lvert s)[r(s, a, s') + \gamma V_{t+1}]$$
- 又根据policy $$\pi$$，直到对于一个state $$s$$，有多个action可以选择，并且optimal的目标是maximize， 并且其中由于我们还不确定planning sequence, 因此我们不能使用$$V_{t+1}$$，但我们可以使用$$V(s')$$来表示经过action a后得到的future reward。则最终得到**Bellman equation**

$$V(s) = \max_{a\in A(s)}\sum_{s'\in S}P_a(s'\lvert s)[r(s, a, s') + \gamma V(s')]= \max_{a\in A(s)}Q(s,a)$$

![image-41](/assets/img/post_img/41.png)

- 我们可以看出，Bellman equation是recursive的。

### 4.3 Policy extraction

$$\pi(s) = arg \max_{a\in A(s)}\sum_{s'\in S}P_a(s'\lvert s)[r(s, a, s') + \gamma V(s')]=arg\max_{a\in A(s)}Q(s, a)$$

对于任意一个state $$s$$，我们选择一个action $$a$$，使得**Q-value**最大。

 

# Partially Observable MDPs

# Value Iteration

![image-42](/assets/img/post_img/42.png)

- Value iteration converges to the **optimal** policy
- 得到optimal policy，我们不需要去获得一个optimal value function $$V$$， 一个**close enough**的value funciton就可以啦。因为small values不会改变最终的policy。
- **Complexity**: 
  - 每个iteration的complexity: $$O(\lvert S\lvert^2\lvert A\lvert)$$
  - 因为每个iteration，outer loop我们需要计算所有$$S$$中的state，接着对于每个state都需要计算$$\sum_{s'\in S}$$，因此需要$$\lvert S\lvert^2$$。并且需要计算每个action对应的value。因此还需要$$\times \lvert A\lvert$$
- the values of states at step $$t+1$$ are dependent only on the value of other states at step $$t$$.

在同一个iteration中，计算结果与选择更新state的顺序无关，由于每个iteration，使用的是上一个iteration得到的V，在同一个iteration更新state reward的时候，不对V进行替换更新，知道一次iteration结束后，将之前的V进行替换。

e.g., 假设在?处向某个方向走发生的概率为0.8，其他方向概率为0.1，且没有reward

![image-43](/assets/img/post_img/43.png)

Right: -0.72

- 0.8 * (0 + 0.9 * (-1)) = -0.72
- 0.1 * (0 + 0.9 * 0) = 0
- 0.1 * (0 + 0.9 * 0) = 0

Left: 0 

up: -0.09

- 0.8 * (0 + 0.9 * 0)  = 0
- 0.1 * (0 + 0.9 * (-1)) = -0.09
- 0.1 * (0 + 0.9 * 0)  = 0

down: -0.09

- 0.8 * (0 + 0.9 * 0)  = 0
- 0.1 * (0 + 0.9 * (-1)) = -0.09
- 0.1 * (0 + 0.9 * 0)  = 0

此时我们选择0，也就是选择撞墙后原地不动

https://colab.research.google.com/github/COMP90054/2022S1_tutorials/blob/master/solution_set_07.ipynb#scrollTo=PLm4V9UO1Oot

### Question1: 计算Q-value

$$Q(s,a)=\sum_{s'\in S}P_a(s'\lvert s)[r(s,a,s')+\gamma V(s')]$$

其中$$S$$表示可从$$s$$通过action $$a$$到达的states

### Question2: 计算value of states

$$V(s)=\max_aQ(s, a)$$

其中$$a$$是所有在state $$s$$处可执行的action

# Policy Iteration

## 1. Basic Idea

The basic idea here is that policy evaluation is easier to computer than value iteration because the set of actions to consider is fixed by the policy that we have so far.

Policy iteration first starts with some (non-optimal) policy, such as a random policy, and then calculates the value of each state of the MDP given that policy — this step is called the *policy evaluation*. It then updates the policy itself for every state by calculating the expected reward of each action applicable from that state.

## 2. Policy Evaluation

calculate expected reward
$$
V^\pi(s) =  \sum_{s' \in S} P_{\pi(s)} (s' \mid s)\ [r(s,a,s') +  \gamma\ V^\pi(s') ]
$$

### 2.1 Optimal expected reward

$$V^*(s)=\max_\pi V^{\pi}(s)$$

### 2.2 Optimal policy

$$\arg \max_\pi V^\pi (s)$$

## 3. Policy improvement

通过更新 actions 基于从policy evaluation得到的$$V(s)$$

$$Q^{\pi}(s,a)  =  \sum_{s' \in S} P_a(s' \mid s)\ [r(s,a,s') \, + \,  \gamma\ V^{\pi}(s')]$$

If there is an action a such that $$Q_\pi(s,a)>Q_\pi(s,\pi(s))$$, then the policy $$\pi$$ can be *strictly improved* by setting $$\pi(s)\leftarrow a$$. This will improve the overall policy.

## 4. Policy Iteration

- policy evaluation + policy improvement
- computes an optimal $$\pi$$ by performing a sequence of interleaved policy evaluations and improvements:

![image-44](/assets/img/post_img/44.png)

- number of policies： $$O(|A|^{|S|})$$
- each iteration costs: $$O(|S|^2 |A| + |S|^3)$$
- 

 