---
layout: post
title: PyTorch
featured-img: PyTorch
summary: 运用PyTorch自定义网络层并搭建网络结构
language: english 
---
##### Table of Contents  
- [mac上安装pytorch](#mac上安装pytorch)  
    - [建立python虚拟环境](#建立python虚拟环境)
    - [激活python虚拟环境](#激活python虚拟环境)
    - [利用pip安装pytorch](#利用pip安装pytorch)
- [搭建自定义网络](#搭建自定义网络)
    - [搭建自定义网络层](#搭建自定义网络层)
    - [自定义网络](#自定义网络)
    - [定义损失函数](#定义损失函数)
    - [定义优化器](#定义优化器)
- [训练网络模型](#训练网络模型)
- [预测结果](#预测结果)
- [例：BP Net](#例：BP Net)

<a name='mac上安装pytorch'/>

# mac上安装pytorch

<a name='建立python虚拟环境'/>

## 建立python虚拟环境
```bash
(base) :~ XXX$ conda create -n my_pytorch python=3.9
```
输入y+Enter
```bash
Proceed ([y]/n)? y
```
```bash
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate my_pytorch
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```
等待出现这些提示，说明已经创建成功

<a name='激活python虚拟环境'/>

## 激活python虚拟环境
```bash
(base) :~ XXX$ conda activate my_pytorch
```
若出现()里面是创建的虚拟环境的名称，则表明已经切换到该虚拟环境下。如下图所示
```bash
(my_pytorch):~ XXX$
```

<a name='利用pip安装pytorch'/>

## 利用pip安装pytorch
```bash
(my_pytorch):~ XXX$ pip3 install torch
```

报错
```bash
pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

出现该错误的原因是当前网速不够，因此选择使用清华镜像进行安装
```bash
(my_pytorch):~ XXX$ python3 -m pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```
```bash
Successfully installed numpy-1.22.1 pillow-9.0.0 torch-1.10.1 torchvision-0.11.2 typing-extensions-4.0.1
```
成功安装

<a name='搭建自定义网络'/>

# 搭建自定义网络

<a name='搭建自定义网络层'/>

## 搭建自定义网络层

- design$$ y = w\times\sqrt{x^2+b}$$

```python
class MyLayer(nn.Module):
    # 初始化：输入输出单元数，权重，偏置
    def __init__(self,in_channel,out_channel,bias=True): # define parameters
        super(MyLayer,self).__init__()#调用父类的构造函数
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.weight=nn.Parameter(torch.Tensor(in_channel,out_channel))#随机产生一个in_channel*out_channel的矩阵权重。又由于权重是要不断训练的，需要将其绑定为一个可以训练的参数于是需要使用Parameter
        if bias:
            self.bias=nn.Parameter(torch.Tensor(in_channel).view(1,-1))#注意⚠️这边的b是自定义层中自带的b，而不是神经网络中的卷积核的偏置，因此维数需要和输入单元数一样
        else:
            self.register_parameter('bias',None)#取消bias这个参数
    def forward(self,x):#前向传播计算
				############################
				# activation function design
				############################
        input_=torch.sqrt(torch.pow(x,2)+self.bias)#相当于卷积层中进行的激活
        output=torch.matmul(input_,self.weight)#矩阵乘法运算 1*in_channel in_channel*out_channel
        return output
```

<a name='自定义网络'/>

## 自定义网络

```python
in_channel1,out_channel1,out_channel2=5,3,1
class MyNet(nn.Module):
    def __init__(self): # define each layer
        super(MyNet,self).__init__()
        self.myLayer1=MyLayer(in_channel1,out_channel1)
    def forward(self,x): # connect each layer
        output=self.myLayer1(x)
        return output
```

<a name='定义损失函数'/>

## 定义损失函数

```python
loss_fn = torch.nn.CrossEntropyLoss() # multi-classification
loss_fn = torch.nn.HingeEmbeddingLoss() # SVM
loss_fn = torch.nn.MSELoss() # Regression
```

<a name='定义优化器'/>

## 定义优化器

```python
learning_rate = 1e-4

optimizer = torch.optim.Adagrad(model.parameters(), lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer = torch.optim.Adadelta(model.parameters(), lr = learning_rate)
```

<a name='训练网络模型'/>

# 训练网络模型

- for each echo
    - step1: predict the labels through **forward**
    - step2: attain error (compare true labels and predicted labels). (loss_fn(y, y_pred))
    - step3: let the gradient as 0. (optimizer.zero_grad())
    - step4: backward the error
    - step5: use optimizer (optimizer.step())

```python
model = MyNet()
x = torch.randn(10,5)  #（10，5）
y = torch.randn(10,3) #（10，3）

for i in range(echos):
    # step1
    y_pred = model(x)
    # step2
    error = loss_fn(y_pred,y)
    # step3
    optimizer.zero_grad()
    # step4
    error.backward()
    # step5
    optimizer.step()
```

<a name='预测结果'/>

# 预测结果

```python
output = model(x)
pre = torch.max(output,1)[1]
pred_y = pre.data.numpy() # data.numpy(): transform tensor to array
target_y = y.data.numpy()
```

<a name='例：BP Net'/>

# 例：BP Net

## Construct BP Net

```python
import torch.nn.functional as fun
class BP(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(BP,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        o = fun.sigmoid(self.hidden(x))
        output = self.out(o)
        return output
```

## Define optimizer and loss function

```python
model = BP(n_feature,n_hidden,n_output)
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
```

## Train model

```python
for i in range(echos):
    output = model(x)
    error = loss_fn(output,y)
    optimizer.zero_grad()
    error.backward()
    optimizer.step()
```

## Get Result

```python
output = model(x)
pre = torch.max(output,1)[1]
pred_y = pre.data.numpy() # data.numpy(): transform tensor to array
target_y = y.data.numpy()
```