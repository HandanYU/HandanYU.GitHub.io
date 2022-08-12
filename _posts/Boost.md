# 基本概念
## 前向分布算法
**基本思想**:每次只学习一个基函数及系数,逐步逼近最优解
## 残差和负梯度的关系
- 残差是负梯度在损失函数是**平方误差**时候的特殊情况

- 对于梯度提升模型我们的目标是寻找一个$$f(x)$$使得损失函数$$L(y,f(x))$$最小。由于是解决**最小化问题**，因此采用**梯度下降**方法，也就是关注**负梯度**

- 损失函数的负梯度可以表示为

$$
-\frac{\partial L(y, f_{m-1}(x))}{\partial f_{m-1}(x)}
$$

- 当损失函数为**平方误差**的时候，也就是

$$
L(y, f_{m-1}(x)) = \frac{1}{2}(y-f_{m-1}(x))^2
$$

则带入上式，得到此时损失函数的负梯度为

$$
-\frac{\partial L(y, f_{m-1}(x))}{\partial f_{m-1}(x)} = y - f_{m-1}(x)
$$

此时我们可以发现$$y - f_{m-1}(x)$$就是当前模型的拟合残差。因此我们通常用以下式子计算残差：

$$
r_{m-1} = -\frac{\partial L(y, f_{m-1}(x))}{\partial f_{m-1}(x)} 
$$

# 梯度提升决策树（Gradient Boosting Decision Tree, GBDT）

- GBDT属于集成学习中的Boosting算法，即是一个串行的算法，通过逐步拟合逼近真实值。
- 其基分类器是回归树(CART)。
- 可以减少bias（误差）却不能减少variance（偏差），因为每次基本都是全样本参与训练，不能消除偶然性的影响，但每次都逐步逼近真实值，可以减少误差。
- 目标：通过寻找新的模型使得残差不断减小。
- 每一棵树学的是之前所有树的残差，这个残差累加后能得到真实值
- 用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART回归树。
## 基学习器——回归树（CART）

$$
T(x;\Theta) = \sum_{j=1}^J c_jI(x\in R_j)
$$

## GBDT模型

$$
\hat{y}^{(0)} = f_0(x) = 0\\
\hat{y}^{(1)} = f_1(x) = T_0(x;\Theta_1) = f_0(x) + T(x;\Theta_1)\\
\hat{y}^{(2)}=f_2(x) = T_1(x;\Theta_1) + T(x;\Theta_2) = f_1(x) + T_2(x;\Theta_2)\\
\cdot\\
\cdot\\
\cdot\\
\hat{y}^{(m)}=f_m(x) = \sum_{j=1}^mT(x;\Theta_m) = f_{m-1}(x) + T_m(x;\Theta_m)
$$

## 损失函数

$$
L(y_i,\hat{y_i})
$$

- 对于回归问题:
    - 常用损失函数有：MAE、MSE、RMSE
    - 当为均方差MAE/平方误差Square Loss的时候： $$L(y_i,\hat{y_i})=(y_i-\hat{y_i})^2$$
- 对于分类问题:
    - 二分类

$$
L(y_i,\hat{y_i})=\log (1+e^{-2y_i\hat{y_i}}),\quad y_i\in\{-1,1\}
$$

- 多分类

$$
L(y_i,\hat{y_i})=-\sum_{k=1}^Ky_k\log \frac{e^{\hat{y_k}}}{\sum_{k=1}^Ke^{\hat{y_k}}}
$$

## 目标函数

我们的目标是【**经验风险最小**】，即

$$
\min_{\Theta_m} \sum_{i=1}^n L(y_i,\hat{y_i}^{(m)})
$$

## 学习过程/求解过程
由于目标是进行最小化，因此通过**梯度下降**的方法来求极值（也就是选择**损失函数的负梯度方向**能够快速找到最优解）

换句话说，也就是通过**损失函数的负梯度方向**去拟合损失值，从而进一步去拟合CART树。用公式表示就是

$$
T_m(x_i;\Theta_m) \approx f_m(x_i) - f_{m-1}(x_i) \\=-\frac{\partial L(y_i,\hat{y_i}^{(m-1)})}{\partial \hat{y_i}^{(m-1)}}
$$

> 另一种理解方法：采用泰勒一阶展开$$f(x+\Delta x) \approx f(x) + f'(x)\Delta x$$

$$
L(y_i,\hat{y_i}^{(m)}) = L(y_i,\hat{y_i}^{(m-1)}+T_m(x_i,\Theta_m))\\
\approx L(y_i,\hat{y_i}^{(m-1)})+\frac{\partial L(y_i,\hat{y_i}^{(m-1)})}{\partial \hat{y_i}^{(m-1)}}T_m(x_i,\Theta_m)
\\\Rightarrow L(y_i,\hat{y_i}^{(m)})-L(y_i,\hat{y_i}^{(m-1)})\approx \frac{\partial L(y_i,\hat{y_i}^{(m-1)})}{\partial \hat{y_i}^{(m-1)}}T_m(x_i,\Theta_m)
$$

由于我们的目的是$$L(y_i,\hat{y_i}^{(m)}) < L(y_i,\hat{y_i}^{(m-1)})$$

则当

$$
T_m(x_i,\Theta_m) = -\frac{\partial L(y_i,\hat{y_i}^{(m-1)})}{\partial \hat{y_i}^{(m-1)}}
$$

的时候一定成立。

## 算法流程步骤
已知有样本数据$${(x_i,y_i)}$$
- Step 1: 取初始学习器:$$f_0(x)$$
- Step 2: 拟合第$$m$$颗回归树
    - 计算残差：将【上一轮的损失函数负梯度(残差)】作为本轮的损失值近似值，
    $$\hat{y_i}^{(m)}=r_{im}=-\frac{\partial L(y_i, f_m(x))}{\partial  f_m(x) }$$
    - 运用残差拟合一个回归树: $$T_m(x;\Theta_m)$$
- Step 3: 拟合函数更新(模型更新)：$$f_m(x) = f_{m-1}(x) + T_m(x;\Theta_m)$$
- Step 4: 不断进行Step2,3。

## 案例