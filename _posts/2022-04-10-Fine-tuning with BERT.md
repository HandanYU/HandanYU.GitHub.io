---
layout: post
title: Fine-tuning with BERT on GPU
summary: 本节主要讲解如何使用pre-trained BERT model来完成sentiment analysis task。（使用PyTorch框架）
featured-img: deep learning
language: chinese
category: NLP
---

# 1. Preprocessing
首先我们来梳理一下**Text Preprocessing**步骤过程。
## 1.1 Tokenisation
在BERT中我们使用的是WordPiece tokenisation，这里我们load预训练好的一个简单的tokenisation model `bert-base-uncased`。
```python
from transformers import BertTokenizer
#load BERT's WordPiece tokenisation model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence = 'I enjoyed this movie sooooo much.'
tokens = tokenizer.tokenize(sentence)
print(tokens)
```

## 1.2 Padding
- 在BERT中，有两个特殊的token（i.e., [CLS], [SEP]）
    - 对于[CLS]，经常加在每个sentence的开头，用于反应这个sentence的信息，并经常用于 fine-tune BERT for downstream tasks
    - 对于[SEP]，经常是加在每个sentence结尾。用于表明这个sentence结束了，开启新的一个sentence。（但当对于single sentence as Input的downstream task，则不需要用到， e.g., sentiment classification）而对于涉及sentence pairs的downstream task (e.g., textual entailment and sentence similarity classification)，则十分需要。**Note:** 由于masked language model需要next-sentence prediction objective，因此也需要[SEP]
- 由于我们在训练model的时候的input是minibatch的形式，因此需要将一个batch中所有sentence的length保持一致，也就是通过padding的方式补全。

```python

# add special tokens
tokens = ['[CLS]'] + tokens + ['[SEP]']
print(tokens)

# padding in order to keep same length
T = 12

padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
print(padded_tokens)

```
## 1.3 Get the attention mask
一个与tokens列表大小一致的列表，其中将`[PAD]`对应位置设为0，其他位置为1
```python
attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
print(attn_mask)
```

## 1.4 Convert tokens into ids
get the ids of each tokens according to the vocabulary
```python
tokens_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
```

# 2. Fine-tuning with BERT using PyTorch frame
## 2.1 Load and process dataset
创建一个Dataset类，用于load data, 
- 其中`__init___` function 中获取raw data和pre-trained model。
- 在`__getitem__` function 中进行之前我们提到的一系列pre-processing，并return我们处理得到的数据
```python
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

class SSTDateset(Dataset):
    def __init__(self, filename, maxlen):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter = '\t')

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        """Triggered when you call dataset[index]
        """

        # get the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']

        # do pre-processing follow the previous contents
        ## Tokenize
        tokens = self.tokenizer.tokenize(sentence)
        ## padding 
        ### add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ### keep the same length
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(toekn))]
        else:
            # Prunning the list to be of specified max length
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']
        ## get indices of tokens in vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ## convert list-type into tensor-type
        tokens_ids_tensor = torch.tensor(tokens_ids)

        ## obtain attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        atten_mask = (tokens_ids_tensor != 0).long() # the id of '[PAD]' in vocab is 0
        return tokens_ids_tensor, atten_mask, label

train_set = SSTDataset(filename='10-train.tsv', maxlen=30)
dev_set = SSTDataset(filename='10-dev.tsv', maxlen=30)

```
## 2.2 Create DataLoader
用PyTorch的DataLoader容器将Dataset包裹起来
```python
train_loader = DataLoader(train_set, batch_size = 64, num_workers = 2)
dev_loader = DataLoader(dev_set, batch_size = 64, num_workers = 2)
```

## 2.3 Create sentiment classifier
```python
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.cls_layer = nn.Linear(768, 1)
    def forward(self, seq_tokens, atten_mask):
        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq_tokens, attention_mask = attn_masks)
        cont_reps = outputs.last_hidden_state
        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        # Feed [CLS] representation to the downstream task: binary classifier
        logits = self.cls_layer(cls_rep)
        return logits
```
## 2.4 Train the model
### 2.4.1 Initialize the model, loss function and optimizer
```python
classifier = SentimentClassifier()
classifier.cuda(gpu) # run on gpu
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr = 2e-5)
```
### 2.4.2 Iteratively train the model
```python
def train(classifier, criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
    st = time.time()
    for ep in range(max_eps):
        classifier.train()
        for iter, (seq_tokens, atten_mask, label) in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()
            # Converting these elements to cuda tensors
            seq_tokens, atten_mask, label = seq_tokens.cuda(gpu), atten_mask.cuda(gpu), label.cuda(gpu)
            # Get prediction
            pred = classifier(seq_tokens, atten_mask)
            # Get loss
            loss = loss_fn(pre.squeeze(-1), label.float())
            # Backpropagating the gradients
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                acc = get_accuracy_from_logits(pred, label)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(iter, ep, loss.item(), acc, (time.time()-st)))
                st = time.time()
        # evaluate on dev-data
        dev_acc, dev_loss = evaluate(classifier, loss_fn, dev_loader, gpu)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(ep, dev_acc, dev_loss))
        if dev_acc > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(classifier.state_dict(), 'sstcls_{}.dat'.format(ep))
```
### 2.4.3 Evaluation dev data
```python
def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc
def evaluate(classifier, loss_fn, dev_loader, gpu):
    classifier.eval()
    mean_acc, mean_loss = 0, 0
    count = 0
    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            pred = classifier(seq, attn_masks)
            mean_loss += loss_fn(pred.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(pred, labels)
            count += 1

    return mean_acc / count, mean_loss / count
```