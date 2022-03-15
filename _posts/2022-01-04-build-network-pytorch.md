---
layout: post
title: Build Network via PyTorch 
featured-img: deep learning
summary: 运用PyTorch自定义网络层并搭建网络结构
language: english 
category: deep learning
---

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

<a name='例：RNN'/>

# 例：RNN
```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm


class RNN(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYER):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(INPUT_SIZE, EMBED_DIM)
        self.rnn = nn.RNN(
            input_size=EMBED_DIM, 
            hidden_size=HIDDEN_SIZE, 
            num_layers=NUM_LAYER, 
            batch_first=True)
        self.out = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE)
    def forward(self, x, h_state, length):
        x = self.embedding(x) # batch_size(2) * seq_len(4) * embed_dim(7)
        rnn_out, h_state = self.rnn(x, h_state) # rnn_out: batch_size(2) * seq_len(4) * hidden_size(2)
        outs = []
        
        for seq in range(rnn_out.size(1)):
            outs.append(self.out(rnn_out[:,seq,:]))
        out = torch.stack(outs, 1)
        return out, h_state # out: batch_size(2) * seq_len(4) * output_size(1)
    


# read txt
content = []
words = []
with open('./text.txt') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        content.append(line.strip())
        words.extend(line.strip().split(' '))
vocab = set(words)

# encode
word_int_table = {w:i for i, w in enumerate(vocab)}
encoded_content = []
for s in content:
    encoded_content.append([word_int_table[w] for w in s.split(' ')])

# pad，在创建batch data的时候要求一个batch中所有instance整齐
seq_len = [len(s) for s in encoded_content]
padded_content = []
for s in encoded_content:
    padded_array = np.pad(np.array(s),(0,max(seq_len)-len(s)),mode='constant')
    padded_content.append(torch.tensor(padded_array))
# concat
x = torch.stack(padded_content, 0)
y = torch.zeros(x.shape)
y[:-1], y[-1] = x[1:], x[0]
x_cat = []
for data_x, label, length in zip(x,y,seq_len):
    x_cat.append((data_x, label,length))


BATCH_SIZE = 2 # 要使得train_loader的大小能被BATCH_SIZE整除，否则对于hidden_state的维数会和最后一个batch的input对应不上
LR = 1e-3

# generate batch data
train_data = DataLoader(x_cat, batch_size=BATCH_SIZE, shuffle=True)

INPUT_SIZE = OUTPUT_SIZE = len(vocab)
EMBED_DIM = 7
HIDDEN_SIZE = 3
NUM_LAYER = 2

model = RNN(INPUT_SIZE, OUTPUT_SIZE, EMBED_DIM, HIDDEN_SIZE, NUM_LAYER)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

h_state = None

    
for epoch in range(10):
    for i, (x,y,length) in enumerate(train_data):
        out, h_state = model(x, h_state, length)
        h_state = Variable(h_state.data) # 注意必须用Variable进行包裹，否则会导致导数计算错误。“one of the variables needed for gradient computation has been modified by an inplace operation”
        loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1).long())
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch: #{}========= loss: {}'.format(epoch,loss))
```

# 保存和加载模型
