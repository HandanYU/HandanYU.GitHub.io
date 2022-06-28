
---
layout: post
title: Model-free reinforcement learning
summary: 本章主要总结了面试中涉及到的机器学习相关的一些问题。
featured-img: machine learning
language: chinese 
category: AI
---

# Model-free reinforcement learning

Learning a policy directly from experience and rewards

 

## 1. Monte-carlo reinforcement learning

The most important is that it has high *variance*.

Recall that we calculate the future discounted reward for an episode and use that to calculate the average reward for each state-action pair.

### 1.1. Q-table

使用**Q Table**来记录Q-function (i.e., Q(s,a))

### 1.2. 时序差分学习 Temporal difference (TD) reinforcement learning

Use *bootstrapping*

![image-48](/assets/img/post_img/48.png)

## 2. Q-learning (off-policy脱离)

an **off-policy** approach，属于TD，运用**Q-function**来estimate **future reward**。

- 记录每个(state, action) pair

- $$\max_{a'}Q(s',a')$$来estimate $$V(s')$$
- 与Monte Carlo Reinforcement leanring不同的是，因为Q-learning使用了**bootstrapped value**， **Q**会在每个step进行update，而不是等到一个episode结束了才update。（早期学习）

### 2.1 **Q-function updation**

![image-49](/assets/img/post_img/49.png)

### 2.2 Q-learning Algorithm 

![image-50](/assets/img/post_img/50.png)



## 3. SARSA (on-policy 依赖)

（State-action-reward-state-action）

SARSA is **on-policy** reinforcement learning

- 用actual $$Q(s', a')$$来estimate $$V(s')$$
- 属于TD learing

![image-51](/assets/img/post_img/51.png)



## 4. Off-policy vs. On-policy

- Off-policy: 通过$$max_{a'}Q(s', a')$$来update，独立于current state
- On-policy：通过actual $$Q(s', a')$$来update，依赖于current state，且next state依赖于前一个state
- 

## 5. Q-learning vs. SARSA

|            | optimal          | irrelevant |      |
| ---------- | ---------------- | ---------- | ---- |
| Q-learning | yes              | Yes        |      |
| SARSA      | no (sub optimal) | no         |      |

SARSA (on-policy) learns action values relative to the policy it follows, while Q-Learning (off-policy) does it relative to the greedy policy.

## 6. Example

![image-52](/assets/img/post_img/52.png)



### Q-learning

假设现在在状态(2,2)，且行动"North"被选择执行了，由于在(2,2)上方没有网格了，因此得到的next state为(2,2)。根据以上Q表，进行一步更新操作。

![image-53](/assets/img/post_img/53.png)

### SARSA

同样由于(2,2)上面没有网格了，只能执行"West"操作，因此

![image-54](/assets/img/post_img/54.png)