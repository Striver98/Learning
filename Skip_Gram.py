# -*- coding:utf-8 -*-
# 指定文件编码为UTF-8，确保中文等特殊字符正确处理
#! usr/bin/env python3
# 指定使用Python3解释器执行此脚本

"""
Created on 10/04/2020 上午12:33
@Author: xinzhi yao
"""
# 文件创建信息和作者信息

# 导入必要的库和模块
import os  # 操作系统接口，用于文件路径操作
import re  # 正则表达式，用于文本处理
import time  # 时间相关功能，用于计时
import string  # 字符串处理，包含标点符号等常量
import random  # 随机数生成
import collections  # 容器数据类型，如deque、Counter等
import numpy as np  # 数值计算库

import matplotlib  # 绘图库
import matplotlib.pyplot as plt  # 绘图功能
from sklearn.manifold import TSNE  # t-SNE降维算法，用于高维数据可视化

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数接口
import torch.optim as optim  # 优化算法

# Step 1: 数据预处理
def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' ', lower=True):
    """
    字符串规范化函数
    参数:
        str_list: 字符串列表
        punc2: 标点符号替换为的字符，默认为空格
        num2: 数字替换为的标记，默认为'NBR'
        space2: 多个空格替换为的字符，默认为单个空格
        lower: 是否转换为小写，默认为True
    返回:
        规范化后的字符串列表
    """
    # 保留连字符，去除其他标点符号
    punctuation = string.punctuation.replace('-', '')
    # 创建输入列表的副本，避免修改原数据
    rep_list = str_list.copy()
    
    # 遍历列表中的每个字符串
    for index, row in enumerate(rep_list):
        row = row.strip()  # 去除字符串首尾的空白字符
        row = re.sub("\d+.\d+", num2, row)  # 替换浮点数为指定标记
        row = re.sub('\d+', num2, row)  # 替换整数为指定标记
        
        # 替换所有标点符号
        for pun in punctuation:
            row = row.replace(pun, punc2)
        
        # 如果需要，转换为小写
        if lower:
            row = row.lower()
        
        # 合并多个连续空格为单个空格
        rep_list[index] = re.sub(' +', space2, row)
    
    return rep_list

def Data_Pre(corpus: str, out: str, head=True):
    """
    数据预处理主函数
    参数:
        corpus: 输入语料文件路径
        out: 输出文件路径
        head: 是否跳过文件首行（标题行），默认为True
    返回:
        输出文件路径
    """
    # 如果输出文件已存在，直接返回路径
    if os.path.exists((out)):
        return out
    
    # 打开输出文件准备写入
    wf = open(out, 'w', encoding='utf-8')
    
    # 打开输入文件读取数据
    with open(corpus, encoding='utf-8') as f:
        if head:
            f.readline()  # 跳过标题行
        
        # 逐行处理文件内容
        for line in f:
            l = line.strip()  # 去除行首尾空白字符
            # 对每行进行规范化处理
            sent_list = str_norm([l], punc2=' ', num2='NBR', space2=' ')
            
            # 将处理后的句子写入输出文件
            for sent in sent_list:
                wf.write('{0}\n'.format(sent))
    
    wf.close()  # 关闭输出文件
    return out

# 原始数据文件路径
raw_file = './data/reference.table.txt'
# 预处理数据，生成规范化后的语料文件
corpus = Data_Pre(raw_file, './data/corpus.txt')

def read_data(filename: str):
    """
    读取数据文件，将所有单词提取到列表中
    参数:
        filename: 输入文件名
    返回:
        包含所有单词的列表
    """
    words = []  # 初始化空列表存储单词
    
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = line.strip().split()  # 分割每行为单词列表
            for word in l:
                words.append(word)  # 将每个单词添加到列表中
    
    return words

# 读取处理后的语料数据
words = read_data((corpus))
# 打印数据大小（单词总数）
print('Data size: {0} words.'.format(format(len(words), ',')))
"""
预期输出示例:
Data size: 3,312 words.
"""

# Step 2: 构建词典并用UNK标记替换稀有词
def build_dataset(words, vocabulary_size=40000):
    """
    构建词汇表并将单词转换为索引
    参数:
        words: 单词列表
        vocabulary_size: 词汇表大小，默认为40000
    返回:
        data: 单词索引列表
        count: 词频统计
        word2idx: 单词到索引的映射字典
        idx2word: 索引到单词的映射字典
    """
    # 初始化词频统计列表，UNK（未知词）计数初始为-1
    token_count = [['UNK', -1]]
    # 添加最常见的vocabulary_size-1个单词及其频次
    token_count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    
    word2idx = dict()  # 创建空字典用于单词到索引的映射
    data = []  # 存储转换后的索引数据
    unk_count = 0  # 未知词计数器
    
    # 构建word2idx字典
    for word, _ in token_count:
        word2idx[word] = len(word2idx)  # 为每个单词分配唯一索引
    
    # 获取词汇表中所有单词的集合
    word_set = set(word2idx.keys())
    
    # 将原始单词列表转换为索引列表
    for word in words:
        if word in word_set:
            index = word2idx[word]  # 单词在词汇表中，使用对应索引
        else:
            index = 0  # 单词不在词汇表中，标记为UNK（索引0）
            unk_count += 1  # 增加UNK计数
        data.append(index)  # 将索引添加到数据列表
    
    # 更新UNK的实际计数
    token_count[0][1] = unk_count
    
    # 创建索引到单词的反向映射字典
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return data, token_count, word2idx, idx2word

# 设置词汇表大小
vocabulary_size = 40000
# 构建数据集
data, count, word2idx, idx2word = build_dataset(words, vocabulary_size)
# 获取词汇表中的所有单词
words = list(word2idx.keys())
# 打印最常见的几个单词
print('Most common words (+UNK)', count[:6])
# 打印样本数据（前10个单词的索引和对应单词）
print('Sample data: index: {0}, token: {1}'.format(data[:10], [idx2word[i] for i in data[:10] ]))
"""
预期输出示例:
Most common words (+UNK) [['UNK', 0], 
 ('of', 554), ('the', 495), ('and', 398), ('in', 392), ('a', 207)]

Sample data: 
index: [792, 1, 5, 128, 129, 17, 556, 3, 793, 1], 
token: ['Cloning', 'of', 'a', 'cDNA', 'encoding', 'an', 'importin-alpha', 'and', 'down-regulation', 'of']
"""

# Step 3: 为skip-gram模型生成训练批次的函数
def generate_batch(data, batch_size, num_skips, skip_window):
    """
    生成Skip-Gram模型训练批次
    参数:
        data: 单词索引列表
        batch_size: 批次大小
        num_skips: 每个中心词生成的上下文词数量
        skip_window: 上下文窗口大小（单侧）
    返回:
        batch: 中心词索引数组
        labels: 上下文词索引数组
    """
    global data_index  # 使用全局变量记录当前数据位置
    
    # 检查参数有效性
    assert batch_size % num_skips == 0  # 批次大小必须是num_skips的倍数
    assert num_skips <= 2 * skip_window  # 采样数量不能超过窗口内可用词数
    
    # 初始化批次和标签数组
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    # 计算总窗口长度（中心词 + 左右各skip_window个词）
    span = 2 * skip_window + 1  # [skip_window个词 中心词 skip_window个词]
    # 创建固定长度的双端队列作为滑动窗口
    buffer = collections.deque(maxlen=span)
    
    # 从data开头添加整个窗口长度的索引到缓冲区
    for _ in range(span):
        buffer.append(data[data_index])  # 将当前索引添加到缓冲区
        # 防止索引溢出，使用取模运算循环使用数据
        data_index = (data_index + 1) % len(data)
        # 调试用打印语句（已注释）
        # print(buffer, '\n')
        # print(data[data_index], idx2word[data[data_index]], '\n')
        """
        预期输出示例:
        deque([852], maxlen=9) 
        1 of 
        deque([852, 1], maxlen=9) 
        5 a 
        deque([852, 1, 5], maxlen=9) 
        144 cDNA 
        deque([852, 1, 5, 144], maxlen=9)
        """
    
    # 生成批次数据
    for i in range(batch_size // num_skips):
        # 中心词在窗口中的位置（中间）
        target = skip_window
        # 需要避免采样的位置列表（初始包含中心词自身）
        targets_to_avoid = [skip_window]
        
        # 为每个中心词生成num_skips个上下文词
        for j in range(num_skips):
            # 随机选择非中心词的上下文位置
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # 随机选择窗口内的位置
                # print(target)  # 调试用
            
            # 将已选位置添加到避免列表
            targets_to_avoid.append(target)
            # print(target,'\t',targets_to_avoid,'\n')  # 调试用
            
            # 将中心词添加到批次
            batch[i * num_skips + j] = buffer[skip_window]
            # 将上下文词添加到标签
            labels[i * num_skips + j, 0] = buffer[target]
        
        # 滑动窗口：添加新词，移除旧词
        buffer.append(data[data_index])
        # 更新数据索引，循环使用数据
        data_index = (data_index + 1) % len(data)
    
    return batch, labels

# 初始化全局数据索引
data_index = 0
# 设置批次大小
batch_size = 16
# 设置上下文窗口大小（左右各4个词）
skip_window = 4
# 设置每个中心词生成的上下文词数量
num_skips = 8

# 生成一个训练批次
batch, labels = generate_batch(data=data, batch_size=batch_size,
                               num_skips=num_skips, skip_window=skip_window)

# 打印生成的训练样本
for i in range(16):
    print(batch[i], idx2word[batch[i]],
          '->', labels[i, 0], idx2word[labels[i, 0]])

"""
预期输出示例:
Cloning of a cDNA 
encoding
an importin-alpha and down-regulation

145 encoding -> 852 Cloning
145 encoding -> 1 of
145 encoding -> 5 a
145 encoding -> 144 cDNA

145 encoding -> 17 an
145 encoding -> 599 importin-alpha
145 encoding -> 3 and
145 encoding -> 853 down-regulation

17 an -> 1 of
17 an -> 5 a
17 an -> 144 cDNA

17 an -> 599 importin-alpha
17 an -> 3 and
17 an -> 853 down-regulation
17 an -> 145 encoding
17 an -> 1 of
"""

# Step 4: 构建Skip-Gram模型
class SkipGram(nn.Module):
    """
    Skip-Gram模型类
    继承自PyTorch的nn.Module
    """
    def __init__(self, args):
        """
        初始化函数
        参数:
            args: 包含模型参数的配置对象
        """
        super().__init__()  # 调用父类初始化方法
        
        # 从参数中获取词汇表大小和嵌入维度
        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size
        
        # 定义嵌入层：将单词索引映射为稠密向量
        # 输入维度：vocabulary_size，输出维度：embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # 注释：W = vd lookup [1*v']*[V*embedding_size] -> [v* embedding_size]
        
        # 定义输出层：将嵌入向量映射回词汇表空间
        # 输入维度：embedding_size，输出维度：vocabulary_size
        self.output = nn.Linear(self.embedding_size, self.vocabulary_size)
        
        # 定义对数softmax层，用于计算概率分布
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        前向传播函数
        参数:
            x: 输入张量，包含中心词索引
        返回:
            log_ps: 对数概率分布
        """
        # x形状: [batch_size]
        x = self.embedding(x)  # 通过嵌入层，x形状变为: [batch_size, embedding_size]
        x = self.output(x)     # 通过输出层，x形状变为: [batch_size, vocabulary_size]
        log_ps = self.log_softmax(x)  # 计算对数概率，形状: [batch_size, vocabulary_size]
        return log_ps

# Step 5: 开始训练
class config():
    """
    训练配置类
    包含模型和训练的各种参数
    """
    def __init__(self):
        # 训练步数
        self.num_steps = 1000
        # 批次大小
        self.batch_size = 128
        # 检查点间隔（每多少步打印一次损失）
        self.check_step = 20
        
        # 词汇表大小
        self.vocabulary_size = 40000
        # 词向量维度
        self.embedding_size = 200  # 嵌入向量的维度
        # 上下文窗口大小
        self.skip_window = 4  # 考虑左右各多少个词
        # 每个输入词生成的标签数量
        self.num_skips = 8  # 重用输入生成标签的次数
        
        # 是否使用GPU
        self.use_cuda = torch.cuda.is_available()
        
        # 学习率
        self.lr = 0.03

# 创建配置对象
args = config()

# 创建Skip-Gram模型实例
model = SkipGram(args)
# 打印模型结构
print(model)
"""
预期输出示例:
SkipGram(
  (embedding): Embedding(40000, 128)
  (output): Linear(in_features=128, out_features=40000, bias=True)
  (log_softmax): LogSoftmax(dim=1)
)
"""

# 如果可用，将模型移动到GPU
if args.use_cuda:
    model = model.to('cuda')

# 定义负对数似然损失函数（适用于对数概率输入）
nll_loss = nn.NLLLoss()
# 定义Adam优化器
adam_optimizer = optim.Adam(model.parameters(), lr=args.lr)

# 打印分隔线和开始训练信息
print('-'*50)
print('Start training.')

# 初始化平均损失和开始时间
average_loss = 0
start_time = time.time()

# 训练循环
for step in range(1, args.num_steps):
    # 生成训练批次
    batch_inputs, batch_labels = generate_batch(
        data, args.batch_size, args.num_skips, args.skip_window)
    
    # 去除标签的单一维度（从[batch_size, 1]变为[batch_size]）
    batch_labels = batch_labels.squeeze()
    
    # 将numpy数组转换为PyTorch张量
    batch_inputs, batch_labels = torch.LongTensor(batch_inputs), torch.LongTensor(batch_labels)
    
    # 如果可用，将数据移动到GPU
    if args.use_cuda:
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
    
    # 前向传播：计算模型输出（对数概率）
    log_ps = model(batch_inputs)
    # 计算损失
    loss = nll_loss(log_ps, batch_labels)
    # 累加损失用于计算平均值
    average_loss += loss
    
    # 梯度清零
    adam_optimizer.zero_grad()
    # 反向传播：计算梯度
    loss.backward()
    # 优化器步进：更新模型参数
    adam_optimizer.step()
    
    # 每隔check_step步打印一次训练信息
    if step % args.check_step == 0:
        end_time = time.time()  # 记录结束时间
        average_loss /= args.check_step  # 计算平均损失
        # 打印当前步数、平均损失和耗时
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(
            step, average_loss, end_time-start_time))
        # 重置开始时间和平均损失
        start_time = time.time()
        average_loss = 0

"""
预期输出示例:
Average loss as step 20: 10.33, cost: 30.53s.
Average loss as step 40: 10.11, cost: 4.31s.
Average loss as step 60: 10.26, cost: 4.27s.
Average loss as step 80: 9.96, cost: 4.28s.
Average loss as step 100: 9.99, cost: 4.23s.
Average loss as step 120: 10.34, cost: 4.40s.
Average loss as step 140: 10.47, cost: 4.26s.
Average loss as step 160: 10.53, cost: 4.29s.
Average loss as step 180: 10.88, cost: 4.33s.
"""

# 训练完成提示
print('Training Done.')
print('-'*50)

# 获取训练后的词嵌入权重
final_embedding = model.embedding.weight.data
# 打印嵌入矩阵的形状
print(final_embedding.shape)
"""
预期输出示例:
torch.Size([40000, 200])
"""

# Step 6: 可视化词嵌入
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """
    使用t-SNE降维后的词嵌入可视化函数
    参数:
        low_dim_embs: 降维后的嵌入向量
        labels: 对应的单词标签
        filename: 保存图像的文件名
    """
    # 检查嵌入向量数量是否足够
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    print('Visualizing.')
    
    # 创建大尺寸图像
    plt.figure(figsize=(18, 18))  # 单位：英寸
    
    # 为每个词绘制点和标签
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]  # 获取点的坐标
        plt.scatter(x, y)  # 绘制散点
        
        # 添加文本标注
        plt.annotate(label,
                     xy=(x, y),  # 标注点位置
                     xytext=(5, 2),  # 文本偏移量
                     textcoords='offset points',  # 文本坐标类型
                     ha='right',  # 水平对齐方式
                     va='bottom')  # 垂直对齐方式
    
    # 保存图像
    plt.savefig(filename)
    print('TSNE visualization is completed, saved in {0}.'.format(filename))

# 设置matplotlib使用非交互式后端（避免图形显示）
matplotlib.use("Agg")

# 创建t-SNE降维器
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# 设置只可视化前500个词
plot_only = 500
# 对词嵌入进行降维
low_dim_embs = tsne.fit_transform(final_embedding[:plot_only, :])
# 获取对应的单词标签
labels = [ idx2word[i ] for i in range(plot_only) ]
# 生成可视化图像
plot_with_labels(low_dim_embs, labels, filename='./data/tsne.png')