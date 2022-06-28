---
layout: post
title: Advanced Model-free reinforcement learning
summary: 本章主要总结了面试中涉及到的机器学习相关的一些问题。
featured-img: machine learning
language: chinese 
category: AI
---

# Q function approximation

 

## 1. Linear Q-learning (Linear Function Approximation)

### 1.1 Key idea

- 运用features和weights的线性组合approximate Q-function

- 不记录所有的情况，而是关注most important things

# Reward shaping

由于TD learning存在弊端，就是rewards are sometimes *sparse*， 因为初始值的问题

## 1. Basic idea

- to give small intermediate rewards to the algorithm that help it converge more quickly.
- We can modify our reinforcement learning algorithm slightly to give the algorithm **some information(*domain knowledge*)** to help, while also guaranteeing optimality.

## 2. Shaped Reward

give some additional *shaped reward* for some actions
$$
Q(s, a)=Q(s, a)+\alpha[r+F(s, s')+\gamma \max_{a'}Q(s', a')-Q(s,a)]\\
G=\sum_{i=0}^\infin \gamma^i(r_i+F(s_i, s_{i+1}))
$$




- 其中$$F(s, s')$$是额外加入的additional reward
  - 当$$F(s, s')>0$$，说明encourage actions that transition from $$s$$ to $$s'$$ in future exploitation.
  - 当$$F(s, s')<0$$，说明discouraging actions that transition like this in future exploitation.
- 我们称$$r+F(s, s')$$为shaped reward for an action
- 我们称$$G$$为shaped reward for the entire trace

## 3. Potential-based Reward Shaping

A well-designed potential function decrease the time to convergence.
$$
F(s, s')=\gamma\Phi(s')-\Phi(s)
$$

- 其中$$\Phi$$为 *potential function* : 相当于heuristic function
- $$\Phi(s)$$表示*potential of state s*

$$
\begin{array}{lll}
G^{\Phi} & = & \sum_{i=0}^{\infty} \gamma^i (r_i + F(s_i,s_{i+1}))\\
         & = & \sum_{i=0}^{\infty} \gamma^i (r_i + \gamma\Phi(s_{i+1}) - \Phi(s_i))\\
         & = & \sum_{i=0}^{\infty} \gamma^i r_i + \sum_{i=0}^{\infty}\gamma^{i+1}\Phi(s_{i+1}) - \sum_{i=0}^{\infty}\gamma^i\Phi(s_i)\\
         & = & G + \sum_{i=0}^{\infty}\gamma^{i}\Phi(s_{i}) - \Phi(s_0) - \sum_{i=0}^{\infty}\gamma^i\Phi(s_i) \\
         & = & G - \Phi(s_0)
\end{array}
$$


$$
Q^{\Phi}(s, a)=Q(s,a)+\Phi(s)
$$

## 4. Q-function initialisation

## 5. Weakness

- A weakness of model-free methods is that they spend a lot of time exploring at the start of the learning. It is not until they find some rewards that the learning begins. This is particularly problematic when rewards are sparse.
- Reward shaping takes in some domain knowledge that “nudges” the learning algorithm towards more positive actions.
- Q-function initialisation is a “guess” of the initial Q-function to guide early exploration
- Reward sharping and Q-function initialisation are equivalent if our potential function is static.

# n-step TD learning

## 1. Idea

The basic idea of n-step reinforcement learning is that we do not update the Q-value immediately after executing an action: we wait n steps and update it based on the n-step return.

## 2. Discounted Future Rewards

$$
G_t=\sum_{i=1}^my^{i-1}r_i\Rightarrow G_t = r_t+\gamma G_{t+1}
$$

当对于TD(0)，像Q-learning, SARSA这种，在更新$$Q(s,a)$$的时候我们不知道$$G_{t+1}$$， 此时我们选择bootstrapping，也就是
$$
G_t=r_t+\gamma V(s_{t+1})
$$

## 3. Truncated Discounted Rewards

$$
G_t^n = r_t+\gamma r_{t+1} + \gamma^2r_{t+2} + \dots \gamma^{n-1} r_{t+n-1} +\gamma^nV(s_{t+n})
$$

In this above expression $$G_t^n$$ is the full reward, *truncated* at $$n$$ steps, at time $$t$$.

## 4. Updating the Q-function

- sum the discounted rewards

  $$G=\sum_{i={\tau +1}}^{\min\{\tau+n, T\}}\gamma^{i-\tau-1}r_i$$

- calculate the n-step expected reward, if $$\tau+n < T$$, that is we are not at the end of the episode, adds the future expect reward 

  $$G=G+\gamma^nQ(s_{\tau_n}, a_{\tau+n})$$

- update the Q-value

  $$Q(s_{\tau}, a_{\tau})=Q(s_{\tau}, a_{\tau})+\alpha[G-Q(s_{\tau}, a_{\tau})]$$

## 5. n-step SARSA

$$
				 G_{\tau}^{n} = \sum_{i = \tau + 1}^{min(\tau + n, T)}\gamma ^{i - \tau - 1}r_{i}\\\textit{If } \tau + n < T \textit{ then } G_{\tau}^{n} = G_{\tau}^{n} + \gamma^{n}Q(S_{\Tau + n}, A_{\tau + n})\\Q(S_{\Tau}, A_{\tau}) = Q(S_{\tau}, A_{\tau}) + \alpha [G_{\tau}^{n} - Q(S_{\Tau}, A_{\tau})]
$$
