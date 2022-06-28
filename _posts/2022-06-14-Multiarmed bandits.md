---
layout: post
title: Multi-armed bandits
summary: 本章主要总结了面试中涉及到的机器学习相关的一些问题。
featured-img: machine learning
language: chinese 
category: AI
---

# Multi-armed bandits

## 1. 平整蒙特卡洛 (Flat Monte Carlo, FMC)

又被称为均匀采样（uniform sampling)

$$Q(s, a) = \frac{1}{N(s,a)}\sum_{t=1}^{N(s)}I_t(s, a)r_t$$

- $$N(s, a)$$是在状态$$s$$下，action $$a$$被执行的次数
- $$N(s)$$表示状态$$s$$被访问的次数
- $$r_t$$是来自状态$$s$$的第$$t$$次模拟得到的reward
- $$I_t(s,a)=1$$,如果状态$$s$$的第$$t$$次模拟中选择了action $$a$$，否则为0

## 2. Exploration 探索 vs. Exploitation 利用

通过均匀分布导致在所有行动中浪费了相同的采样时间，我们可以通过考虑最有前景的行为，也就是那些好的行动。

因此，我们只要继续执行当前可以得到最佳回报的行动就行。注意：我们的选择仍然是随机的。

### 2.1 The Fear of Missing Out (FOMO)

寻找一种**最小化后悔(regret)**的策略$$\pi$$

$$R(\pi, t)=t\max_a Q^*(a)-E[\sum_{k=1}^tX_{\pi(k),k}]$$

- 其中$$Q^*(a)$$是执行$$a$$的平均真实回报，但我们是不知道具体值的。
- 如果我们执行$$max_aQ^*(a)$$，也就是执行最佳行动，则是没有遗憾的，也就是regret=0。
- 但如果我们没有执行最佳行动，而是根据$$\pi$$执行了行动$$b$$,则
  - $$R$$=执行最佳行动的回报 - 执行行动$$b$$的回报

## 3. Solutions for minimising regret/multi-armed bandit algorithms

### 3.1 $$\epsilon$$-greedy strategy

- Balance exploration and exploitation
- $$\epsilon\in[0,1]$$ controls how much we explore and how much we exploit
- 一般情况下，$$\epsilon\in $$0.05~0.1 work well as they exploit what they have learnt, while still exploring.

#### **Selection mechanism: how to choose an action**

- with probability $$1-\epsilon$$, 选择$$\arg \max_a Q(a)$$，当多个actions都有largest Q-value，break the tie randomly
- with probability $$\epsilon$$， randomly choose with uniform probability.

#### $$\epsilon$$ 大小

![image-45](/assets/img/post_img/45.png)

- **higher values** of **epsilon** tend to have a **lower reward** over time, except that we need some non-zero value of epsilon
- **higher values** mean **more exploration**, so the bandit **spends more time** exploring less valuable actions, even after it has a good estimate of the value of actions. 

### 3.2 $$\epsilon$$-decreasing strategy

- 在$$\epsilon$$-greedy strategy 基础上引入了“decay” $$\alpha\in[0,1]$$，用来decrease $$\epsilon$$。

#### **Selection mechanism**

与$$\epsilon$$-greedy strategy类似，只是每次迭代$$\epsilon=\epsilon \times \alpha$$

- initially, 设置higher value of $$\epsilon$$
- $$\epsilon$$慢慢的被decay变小，同时获得更多的reward

#### $$\alpha$$ 大小

![image-46](/assets/img/post_img/46.png)

- $$\alpha$$过小，会容易很早使得$$\epsilon$$降低到0
- $$\alpha$$的选择取决于具体问题和length of episode，longer episode, decreasing slower更好些，也就是$$\alpha$$大一些

### 3.3 Softmax strategy

**Exploit actions proportionally based on their Q-value**

**在drift情况下，表现**

#### Selection mechanism

选择某个action的概率取决于它对应的**Q-value**

**Boltzman distribution**: $$\frac{e^{Q(a)/\tau}}{\sum_{b=1}^Ne^{Q(b)/\tau}}$$

- $$N$$： the number of arms
- $$\tau$$: $$\tau >0$$, *temperature*， 表示了past data对decision的影响程度
  - **higher value** of $$\tau$$, 意味着选择一个action的probability和其他任意action很**接近**，因为当$$\tau$$接近于infinity的时候，softmax方法接近于uniform strategy
  - **lower value** of $$\tau$$，the probabilities are closer to their Q values.
  - 当$$\tau = 0$$，softmax approaches a **greedy strategy**.

- 同样类似$$\epsilon$$-greedy strategy，可以通过引入decay $$\alpha$$， allows the value of $$\tau$$ to decay until it reaches 1
  - high $$\tau$$ (**more exploration**) in earlier phases, 
  - and low $$\tau$$ (**exploration less**) as we gather more feedback.

#### $$\tau$$ 大小

![image-47](/assets/img/post_img/47.png)

### 3.4 置信区间上界 Upper Confidence Bounds (UCB1)

一种更高效的Multi-armed bandits 策略

#### Selection mechanism

$$
\pi(s) = \arg\max_a Q(a)+\sqrt{\frac{2\ln t}{N(a)}}
$$

- $$t$$是 the number of rounds，也就是状态$$s$$被访问次数
- $$N(a)$$表示在previous rounds中action $$a$$被执行次数
- 可以看出上面的式子不支持$$N(a)=0$$，为了避免这一现象出现，前N轮每个bandit都选择一次
- 加号左边部分$$Q(a)$$，用于鼓励exploitation：对于具有较高reward的action，$$Q$$-value会比较高
- 加号右边部分$$\sqrt{\frac{2\ln t}{N(a)}}$$，用于鼓励exploration：对于被较少exploration的action（也就是N(a)较小），改值会比较高，



# 蒙特卡洛树搜索 （Monte Carlo Tree Search, MCTS）

MCTS 通常用于 **在线决策（online decision）** 或者 **在线学习（online learning）**。agent 在每次访问一个新状态时都会调用 MCTS。

#### online learning vs. offline learning

- 在离线学习中，我们提前完成所有的规划，当执行行动的时候，我们执行之前规划的所有行动，例如：在经典规划中，我们先规划好了一个行动序列，然后依次执行它们，我们也可能从中提取策略，并按照策略行动。
- 在在线学习中，行动和选择是交错进行的，例如：在 MCTS 中，我们在每一次规划都会用来选择下一个行动，然后在执行该行动后，我们将重新进行下一轮的规划。
- 价值评估将提供一个完全收敛的策略，它更适用于离线规划；MCTS 则更倾向于在线规划；而经典规划中的启发式搜索既可用于在线学习，也可用于离线学习。

### 1. 基本特征

- 每个状态的价值 V(s) 通过 **随机模拟（random simulation）**来近似。
- ExpectiMax 搜索树是增量式地(incrementally)构建的。
- 当一些预定义的计算预算用完时（例如：超出时间限制或扩展的结点数），搜索将终止。因此，它是一种 **任意时间*anytime*** 算法，因为它可以随时终止并且仍然给出一个答案。
- 算法将返回表现最好的行动。
  - **complete** if there are *no* dead–ends.
  - **optimal** if an entire search can be performed

## 2. 框架

### 2.1 Selection

在树中选择一个 **未完全扩展** 的单结点。这意味着它至少有一个子结点尚未被探索

### 2.2 Expansion

从一个节点出发，执行一个available action，然后expand其对应的子节点

### 2.3 Simulation

从expanded得到的一个子节点出发，进行随机模拟直到到达terminating state。

### 2.4 Backpropagate

将terminating state的值反馈到root node，通过一步一步更新ancestor node的value。
$$
N(s,a) \leftarrow N(s,a) + 1\\
Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}[r + \gamma G - Q(s,a)]\\
G \leftarrow r + G\\
$$
**Note**: Because action outcomes are selected according to $$P_a(s'\lvert s)$$, this will converge to the average expected reward. This is why the tree is called an *ExpectiMax* tree: we maximise the expected return.

- **But:** what if we do not know $$P_a(s'\lvert s)$$?

Provided that we can *simulate* the outcomes; e.g. using a code-based simulator, then this does not matter. Over many simulations, the Select (and Expand/Execute steps) will sample $$P_a(s'\lvert s)$$ sufficiently close that Q(s,a) will converge to the average expected reward. Note that this is **not** a **model-free** approach: we still need a model in the form of a simulator, but we do not need to have explicit tranisition and reward functions.

