import numpy as np
import torch
import random
from torch.utils import data
from d2l import torch as d2l

#根据带有噪声的线性模型构造一个人造数据集，y为真实值
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

#生成散点图
#d2l.set_figsize()
#d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
#d2l.plt.show()

def load_array(data_arrays,batch_size,is_train=True):
    """构造一个PyTorch数据迭代器，加载数据"""
    dataset = data.TensorDataset(*data_arrays)#将多个张量组合成一个数据集
    return data.DataLoader(dataset,batch_size,shuffle= is_train)#加载（获取）数据
#按批次加载数据
batch_size = 15
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))#获取第一个批次的数据

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # 线性模型。
#nn.Sequential是一个容器模块，可以将多个层按顺序组合在一起。适用于简单的前馈神经网络
#nn.Linear(2, 1)  
#2: 输入特征的数量（特征的维度）（features[0]一行两列）
#1: 输出特征的数量（预测的目标值维度）。在这里，它意味着模型将产生一个单一的输出值
net[0].weight.data.normal_(0, 0.01)  # 初始化权重
net[0].bias.data.fill_(0)  # 初始化偏置
loss = nn.MSELoss()  # MSE计算预测值与真实值之间的平方差的均值

trainer = torch.optim.SGD(net.parameters(), lr=0.03) 
#随机梯度下降（SDG）优化器，学习率 LearningRate=0.03

num_epochs = 10 #三个训练轮次
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)  # 计算当前批次的损失，net（X）计算预测值
        trainer.zero_grad()  
        #先清空之前的梯度。这是因为PyTorch会累加梯度，如果不清空，可能会导致错误的梯度计算
        l.backward()  # 反向传播
        trainer.step()  # 更新参数
    l = loss(net(features), labels)  # 计算整个数据集的损失，有助于监控模型的训练进展
    print(f'epoch {epoch + 1}, loss {l:f}')
# 打印当前epoch的编号和对应的损失值。这有助于追踪模型的训练过程，观察损失是否在逐渐减少