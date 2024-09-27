import numpy as np
import torch
import random
from torch.utils import data
from d2l import torch as d2l

#根据带有噪声的线性模型构造一个人造数据集
def synthetic_data(w,b,num_examples):
        # y = Xw + b + 噪声
    X = torch.normal(0,1,(num_examples,len(w)))  # normal高斯分布
    # （均值，标准差，（样本量，样本长度））
    y = torch.matmul(X,w) + b # matmul矩阵相乘
    y += torch.normal(0,0.01,y.shape) #均值为0，标准差为0.01，与y形状相同的随机噪音
    return X, y.reshape((-1,1)) #返回X,y的列向量

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)

#feature中每一行都包含一个二维数据样本，labels中的每一行都包含一维标签值（一个标量）
print('features:',features[0],'\nlabels:',labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1);
d2l.plt.show()

#定义一个data_iter函数，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)#打乱下标
    for i in range(0,num_examples,batch_size):
        batch_size = torch.tensor(indices[i:min(i + batch_size,num_examples)])
        yield features[batch_size], labels[batch_size]

batch_size = 10

for X,y in data_iter(batch_size, features,labels):
    print(X,'\n',y)
    break