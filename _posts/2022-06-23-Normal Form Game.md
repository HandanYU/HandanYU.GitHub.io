---
layout: post
title: Normal Form Games
summary: 本章主要总结了面试中涉及到的机器学习相关的一些问题。
featured-img: machine learning
language: chinese 
category: AI
---

# Normal form games

![image-55](/assets/img/post_img/55.png)


## 1. Solutions for normal form games（Strategy)

### 1.1 Pure strategy

agent只选择**一个**action并执行它。并且多次执行该游戏，每次使用的都是同一个action。 （$S_i=A_i$）

#### **Expected utility of a pure strategy**

for $$a_i\in A_i$$, $$U_i(a_i)=p_1\times u_i(a_i, a_{-i}^1)+p_2\times u_i(a_i, a_{-i}^2)+\dots +p_m\times u_i(a_i, a_{-i}^m)$$

#### **Indifference**

- an agent $$i$$ is **indifferent** between a set of **pure strategies** $$I\sube A_i$$, if $$\forall a_i, a_j\in I$$, we have $$U_i(a_i)=U_i(a_j)$$

- 也就是如果所有的expected strategies得到的utility都是一样的，那么称该agent是indifferent between pure strategies (it doesn't matter which action they choose)

### 1.2 Mixed strategy

- agent每次根据actions的probability distribution (e.g., 0.8->a, 0.2->b)，选择一个action。也就是当agent进行该游戏需要进行n次action，则会有0.8\*n次选择a， 0.2\*n次选择b

- 当某个action的probability=1的时候，转化为pure strategy
- 当一个游戏中如果每次都执行同一个strategy，则对方很容易beat。因此每次需要随机的选取strategy





### 1.3 Dominant strategy

- **strategy profile**: 一个agent可以执行的strategy
  - 令$S_i$表示agent $$i$$的strategy profile，$$S_{-i}$$表示除了agent $i$以外的agents的strategy profile
  - **Note**: $$S_i\neq A_i$$， 因为strategy有可能是mixed strategy
  - mixed-strategy profiles for all agents $$S=S_1\times S_2\times \dots \times S_n$$
- $$s_i$$ **weakly/strictly** dominates $$s_i'$$ iff and only if $$\forall s_{-i}\in S_{-i}, u_i(s_i, s_{-i}) \geq u_i(s'_i, s_{-i})$$

  - 若使得某个strategy是agent1的weakly strategy，只需要满足该strategy第一个元素在每列中都是最大值或与其他相等

  - 若使得某个strategy是agent**2**的weakly strategy，只需要满足该strategy第**二**个元素在每**行**中都是最大值或与其他相等

- $$s_i$$ **strongly/strictly** dominates $$s_i'$$ iff and only if $$\forall s_{-i}\in S_{-i}, u_i(s_i, s_{-i}) > u_i(s'_i, s_{-i})$$

  - 
- **weakly (strictly) dominant strategy**: if the strategy weakly (strictly) dominates all other strategies.

If all players have a *strictly* dominant strategy in a game, then there exists a unique Nash equilibria.

However, if at least one player has only a dominant (or *weakly dominant)* strategy, then there may be multiple Nash equilibria.

- Prisoner’s dilemma 的strategy属于 strictly dominant的


## 2. Best response

### 2.1 Definition

- from the perspective of one of the agents
- agent's best response: the best strategy that an agent could select *if* it know how all of the other agents in the game were going to play
- **Best response**: 对于agent $$i$$， 如果它的对手执行了strategy profile $s_{-i}\in S_{-i}$是一个*mixed strategy* （即$s_i^*\in S_i$ 满足$u(s_i^*,s_{-i})\geq u(s'_i, s_{-i})$），且所有的strategies $s_i'\in S_i$. 
- 通常情况下，会有多个best response同时存在

### 2.2 Calculation

- complexity O($\lvert S_i\lvert$)
- 

## 3. **Nash** **equilibrium **纳什均衡

### 3.1 Definition

- 所有的有限(finite) normal form games有一个Nash equilibrium
- 通俗易懂的理解Nash equilibrium就是：这是一个*stable strategy*，也就是当所有的agents都保持它们的strategy不变，则每个agent都没有改变strategy的动力激励
- 当所有的agents $i$ 与其对应的所有strategies $s_i$ 是best response to the strategy $s_{-i}$的，则我们称该strategy profile $s=(s_1, \dots, s_n)$为**Nash equilibrium**
- 当其中的strategies $s_i$是pure strategy，则称该strategy profile为**pure-strategy Nash equilibrium**
- 当其中的strategies $s_i$是mixed strategy，则称该strategy profile为**mixed-strategy Nash equilibrium**

### 3.2 Calculation

在所有的strategy profiles中搜索，得到某个strategy profile满足所有agent strategies都是best response

简单来说，对于一个normal form game，我们可以通过遍历game的每个cell，然后对于这个cell，搜索它所在行所在列，判断是否有其他player能够通过改变它们的strategy(保持其他agent的strategy不变）做的结果保持不变或者更好，则说明这个cell中的value是nash equilibrium

#### Example

![image-57](/assets/img/post_img/57.png)


对于一个单元格(x, y)，

- 首先判断x是否符合，则看该列中所有x处的最大值，若当前x不是最大值，则不符合。
- 若x不符合直接判断它不是nash equilibrium
- 若x符合，则看y，看该单元格所在行y处的最大值，若当前y不是最大值则不符合，若是最大值则符合

e.g （method1).,

- (split, split) 坐标为(1, 1): 
  - 首先看第一列中的第一个元素，由于2才是最大的，因此直接判断它不是nash equilibrium
- (split, steal) 坐标为(1, 2):
  - 首先看第二列中的第一个元素，由于0就是最大的了，因此符合
  - 在看第一行中的第二个元素，由于2就是最大的了，因此符合
- (steal, steal) 坐标(2, 2):
  - 首先看第二列中的第一个元素，由于0就是最大的了，因此符合
  - 再看第二行中的第二个元素，由于2才是最大的，因此不符合

e.g. (method2)

- (split, split): 
  - 对于agent1，若其执行steal，则变为(steal, split)， agent1的utility从1->2，因此可以判断(split, split)不是nash equilibrium
- (split, steal)
  - 对于agent1，若其执行了steal，则变为(steal, steal)，agent1的utility从0->0，符合
  - 再看agent2，若其执行了split， 则变为(split, split)， agent2的utility从2->1，符合
  - 则说(split, steal)是nash equilibrium

### 3.3 **Mixed-strategy** **Nash** **equilibria**

- 对每个agent $i\in N$ 都有一个probabilities $P_i=(p_1, \dots, p_m)$，且$\sum_{i=1}^m p_i=1$。并且所有其他opponents $$j$$ are **indifferent** to their **pure strategies** $$A_j$$.
- each agent should choose a mixed strategy such that it makes their opponents indifferent to their own actions.
  - 因为如果我方选择的mixed strategy不能使得opponents **indifferent**，也就意味着opponents那边至少有一个strategy能够得到higher utility，那么这个opponents肯定会选择这个strategy来beat我方
  - 因此为了使得我方更有胜算，因此需要使得opponents不能一目了然的选择strategy
  - 另外选择mixed strategy其实就是选择一个probabilities集合

#### Example

![image-56](/assets/img/post_img/56.png)
				

- 若adversary想要使得defender **indifferent** between its two pure strategies

  - 假设选择Terminal1的probability=Y，则选择Terminal2的probability=1-Y
  - 首先写出defender的所有strategies的utility表达式
    - $$U_D(T1)= 5Y -(1-Y)=6Y-1$$
    - $$U_D(T2)=-5Y+2(1-Y)=-7Y+2$$ 
  - 为了满足indifferent
    - $$U_D(T1)=U_D(T2) \Rightarrow 6Y-1=-7Y+2 \Rightarrow Y=\frac{3}{13}$$

  - 也就是adversary选择Terminal1的概率为$$\frac{3}{13}$$, Terminal2的概率为$$\frac{10}{13}$$

- 若defender想要使得adversary **indifferent** between its two pure strategies

  - 假设选择Terminal1的probability=Y，则选择Terminal2的probability=1-Y
  - 首先写出adversary的所有strategies的utility表达式
    - $$U_A(T1)=-3Y+5(1-Y)=-8Y+5$$
    - $$U_A(T2)=Y-(1-Y)=2Y-1$$
  - 为了满足indifferent
    - $$U_A(T1)=U_A(T2)\Rightarrow -8Y+5=2Y-1\Rightarrow Y=\frac{3}{5}$$
  - 也就是defender选择Terminal1的概率为$$\frac{3}{5}$$， 选择Terminal2的概率为$$\frac{2}{5}$$

# Extensive form games

an extensive form game can be solved with **model-free reinforcement learning techniques** and **Monte-Carlo tree search techniques**: we can just treat our opponent as the environment; albeit one that has a particular goal itself.