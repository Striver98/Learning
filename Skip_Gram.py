# -*- coding:utf-8 -*-
# ָ���ļ�����ΪUTF-8��ȷ�����ĵ������ַ���ȷ����
#! usr/bin/env python3
# ָ��ʹ��Python3������ִ�д˽ű�

"""
Created on 10/04/2020 ����12:33
@Author: xinzhi yao
"""
# �ļ�������Ϣ��������Ϣ

# �����Ҫ�Ŀ��ģ��
import os  # ����ϵͳ�ӿڣ������ļ�·������
import re  # ������ʽ�������ı�����
import time  # ʱ����ع��ܣ����ڼ�ʱ
import string  # �ַ����������������ŵȳ���
import random  # ���������
import collections  # �����������ͣ���deque��Counter��
import numpy as np  # ��ֵ�����

import matplotlib  # ��ͼ��
import matplotlib.pyplot as plt  # ��ͼ����
from sklearn.manifold import TSNE  # t-SNE��ά�㷨�����ڸ�ά���ݿ��ӻ�

import torch  # PyTorch���ѧϰ���
import torch.nn as nn  # ������ģ��
import torch.nn.functional as F  # �����纯���ӿ�
import torch.optim as optim  # �Ż��㷨

# Step 1: ����Ԥ����
def str_norm(str_list: list, punc2=' ', num2='NBR', space2=' ', lower=True):
    """
    �ַ����淶������
    ����:
        str_list: �ַ����б�
        punc2: �������滻Ϊ���ַ���Ĭ��Ϊ�ո�
        num2: �����滻Ϊ�ı�ǣ�Ĭ��Ϊ'NBR'
        space2: ����ո��滻Ϊ���ַ���Ĭ��Ϊ�����ո�
        lower: �Ƿ�ת��ΪСд��Ĭ��ΪTrue
    ����:
        �淶������ַ����б�
    """
    # �������ַ���ȥ������������
    punctuation = string.punctuation.replace('-', '')
    # ���������б�ĸ����������޸�ԭ����
    rep_list = str_list.copy()
    
    # �����б��е�ÿ���ַ���
    for index, row in enumerate(rep_list):
        row = row.strip()  # ȥ���ַ�����β�Ŀհ��ַ�
        row = re.sub("\d+.\d+", num2, row)  # �滻������Ϊָ�����
        row = re.sub('\d+', num2, row)  # �滻����Ϊָ�����
        
        # �滻���б�����
        for pun in punctuation:
            row = row.replace(pun, punc2)
        
        # �����Ҫ��ת��ΪСд
        if lower:
            row = row.lower()
        
        # �ϲ���������ո�Ϊ�����ո�
        rep_list[index] = re.sub(' +', space2, row)
    
    return rep_list

def Data_Pre(corpus: str, out: str, head=True):
    """
    ����Ԥ����������
    ����:
        corpus: ���������ļ�·��
        out: ����ļ�·��
        head: �Ƿ������ļ����У������У���Ĭ��ΪTrue
    ����:
        ����ļ�·��
    """
    # �������ļ��Ѵ��ڣ�ֱ�ӷ���·��
    if os.path.exists((out)):
        return out
    
    # ������ļ�׼��д��
    wf = open(out, 'w', encoding='utf-8')
    
    # �������ļ���ȡ����
    with open(corpus, encoding='utf-8') as f:
        if head:
            f.readline()  # ����������
        
        # ���д����ļ�����
        for line in f:
            l = line.strip()  # ȥ������β�հ��ַ�
            # ��ÿ�н��й淶������
            sent_list = str_norm([l], punc2=' ', num2='NBR', space2=' ')
            
            # �������ľ���д������ļ�
            for sent in sent_list:
                wf.write('{0}\n'.format(sent))
    
    wf.close()  # �ر�����ļ�
    return out

# ԭʼ�����ļ�·��
raw_file = './data/reference.table.txt'
# Ԥ�������ݣ����ɹ淶����������ļ�
corpus = Data_Pre(raw_file, './data/corpus.txt')

def read_data(filename: str):
    """
    ��ȡ�����ļ��������е�����ȡ���б���
    ����:
        filename: �����ļ���
    ����:
        �������е��ʵ��б�
    """
    words = []  # ��ʼ�����б�洢����
    
    with open(filename, encoding='utf-8') as f:
        for line in f:
            l = line.strip().split()  # �ָ�ÿ��Ϊ�����б�
            for word in l:
                words.append(word)  # ��ÿ��������ӵ��б���
    
    return words

# ��ȡ��������������
words = read_data((corpus))
# ��ӡ���ݴ�С������������
print('Data size: {0} words.'.format(format(len(words), ',')))
"""
Ԥ�����ʾ��:
Data size: 3,312 words.
"""

# Step 2: �����ʵ䲢��UNK����滻ϡ�д�
def build_dataset(words, vocabulary_size=40000):
    """
    �����ʻ��������ת��Ϊ����
    ����:
        words: �����б�
        vocabulary_size: �ʻ���С��Ĭ��Ϊ40000
    ����:
        data: ���������б�
        count: ��Ƶͳ��
        word2idx: ���ʵ�������ӳ���ֵ�
        idx2word: ���������ʵ�ӳ���ֵ�
    """
    # ��ʼ����Ƶͳ���б�UNK��δ֪�ʣ�������ʼΪ-1
    token_count = [['UNK', -1]]
    # ��������vocabulary_size-1�����ʼ���Ƶ��
    token_count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    
    word2idx = dict()  # �������ֵ����ڵ��ʵ�������ӳ��
    data = []  # �洢ת�������������
    unk_count = 0  # δ֪�ʼ�����
    
    # ����word2idx�ֵ�
    for word, _ in token_count:
        word2idx[word] = len(word2idx)  # Ϊÿ�����ʷ���Ψһ����
    
    # ��ȡ�ʻ�������е��ʵļ���
    word_set = set(word2idx.keys())
    
    # ��ԭʼ�����б�ת��Ϊ�����б�
    for word in words:
        if word in word_set:
            index = word2idx[word]  # �����ڴʻ���У�ʹ�ö�Ӧ����
        else:
            index = 0  # ���ʲ��ڴʻ���У����ΪUNK������0��
            unk_count += 1  # ����UNK����
        data.append(index)  # ��������ӵ������б�
    
    # ����UNK��ʵ�ʼ���
    token_count[0][1] = unk_count
    
    # �������������ʵķ���ӳ���ֵ�
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return data, token_count, word2idx, idx2word

# ���ôʻ���С
vocabulary_size = 40000
# �������ݼ�
data, count, word2idx, idx2word = build_dataset(words, vocabulary_size)
# ��ȡ�ʻ���е����е���
words = list(word2idx.keys())
# ��ӡ����ļ�������
print('Most common words (+UNK)', count[:6])
# ��ӡ�������ݣ�ǰ10�����ʵ������Ͷ�Ӧ���ʣ�
print('Sample data: index: {0}, token: {1}'.format(data[:10], [idx2word[i] for i in data[:10] ]))
"""
Ԥ�����ʾ��:
Most common words (+UNK) [['UNK', 0], 
 ('of', 554), ('the', 495), ('and', 398), ('in', 392), ('a', 207)]

Sample data: 
index: [792, 1, 5, 128, 129, 17, 556, 3, 793, 1], 
token: ['Cloning', 'of', 'a', 'cDNA', 'encoding', 'an', 'importin-alpha', 'and', 'down-regulation', 'of']
"""

# Step 3: Ϊskip-gramģ������ѵ�����εĺ���
def generate_batch(data, batch_size, num_skips, skip_window):
    """
    ����Skip-Gramģ��ѵ������
    ����:
        data: ���������б�
        batch_size: ���δ�С
        num_skips: ÿ�����Ĵ����ɵ������Ĵ�����
        skip_window: �����Ĵ��ڴ�С�����ࣩ
    ����:
        batch: ���Ĵ���������
        labels: �����Ĵ���������
    """
    global data_index  # ʹ��ȫ�ֱ�����¼��ǰ����λ��
    
    # ��������Ч��
    assert batch_size % num_skips == 0  # ���δ�С������num_skips�ı���
    assert num_skips <= 2 * skip_window  # �����������ܳ��������ڿ��ô���
    
    # ��ʼ�����κͱ�ǩ����
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
    # �����ܴ��ڳ��ȣ����Ĵ� + ���Ҹ�skip_window���ʣ�
    span = 2 * skip_window + 1  # [skip_window���� ���Ĵ� skip_window����]
    # �����̶����ȵ�˫�˶�����Ϊ��������
    buffer = collections.deque(maxlen=span)
    
    # ��data��ͷ����������ڳ��ȵ�������������
    for _ in range(span):
        buffer.append(data[data_index])  # ����ǰ������ӵ�������
        # ��ֹ���������ʹ��ȡģ����ѭ��ʹ������
        data_index = (data_index + 1) % len(data)
        # �����ô�ӡ��䣨��ע�ͣ�
        # print(buffer, '\n')
        # print(data[data_index], idx2word[data[data_index]], '\n')
        """
        Ԥ�����ʾ��:
        deque([852], maxlen=9) 
        1 of 
        deque([852, 1], maxlen=9) 
        5 a 
        deque([852, 1, 5], maxlen=9) 
        144 cDNA 
        deque([852, 1, 5, 144], maxlen=9)
        """
    
    # ������������
    for i in range(batch_size // num_skips):
        # ���Ĵ��ڴ����е�λ�ã��м䣩
        target = skip_window
        # ��Ҫ���������λ���б���ʼ�������Ĵ�����
        targets_to_avoid = [skip_window]
        
        # Ϊÿ�����Ĵ�����num_skips�������Ĵ�
        for j in range(num_skips):
            # ���ѡ������Ĵʵ�������λ��
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # ���ѡ�񴰿��ڵ�λ��
                # print(target)  # ������
            
            # ����ѡλ����ӵ������б�
            targets_to_avoid.append(target)
            # print(target,'\t',targets_to_avoid,'\n')  # ������
            
            # �����Ĵ���ӵ�����
            batch[i * num_skips + j] = buffer[skip_window]
            # �������Ĵ���ӵ���ǩ
            labels[i * num_skips + j, 0] = buffer[target]
        
        # �������ڣ�����´ʣ��Ƴ��ɴ�
        buffer.append(data[data_index])
        # ��������������ѭ��ʹ������
        data_index = (data_index + 1) % len(data)
    
    return batch, labels

# ��ʼ��ȫ����������
data_index = 0
# �������δ�С
batch_size = 16
# ���������Ĵ��ڴ�С�����Ҹ�4���ʣ�
skip_window = 4
# ����ÿ�����Ĵ����ɵ������Ĵ�����
num_skips = 8

# ����һ��ѵ������
batch, labels = generate_batch(data=data, batch_size=batch_size,
                               num_skips=num_skips, skip_window=skip_window)

# ��ӡ���ɵ�ѵ������
for i in range(16):
    print(batch[i], idx2word[batch[i]],
          '->', labels[i, 0], idx2word[labels[i, 0]])

"""
Ԥ�����ʾ��:
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

# Step 4: ����Skip-Gramģ��
class SkipGram(nn.Module):
    """
    Skip-Gramģ����
    �̳���PyTorch��nn.Module
    """
    def __init__(self, args):
        """
        ��ʼ������
        ����:
            args: ����ģ�Ͳ��������ö���
        """
        super().__init__()  # ���ø����ʼ������
        
        # �Ӳ����л�ȡ�ʻ���С��Ƕ��ά��
        self.vocabulary_size = args.vocabulary_size
        self.embedding_size = args.embedding_size
        
        # ����Ƕ��㣺����������ӳ��Ϊ��������
        # ����ά�ȣ�vocabulary_size�����ά�ȣ�embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # ע�ͣ�W = vd lookup [1*v']*[V*embedding_size] -> [v* embedding_size]
        
        # ��������㣺��Ƕ������ӳ��شʻ��ռ�
        # ����ά�ȣ�embedding_size�����ά�ȣ�vocabulary_size
        self.output = nn.Linear(self.embedding_size, self.vocabulary_size)
        
        # �������softmax�㣬���ڼ�����ʷֲ�
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        ǰ�򴫲�����
        ����:
            x: �����������������Ĵ�����
        ����:
            log_ps: �������ʷֲ�
        """
        # x��״: [batch_size]
        x = self.embedding(x)  # ͨ��Ƕ��㣬x��״��Ϊ: [batch_size, embedding_size]
        x = self.output(x)     # ͨ������㣬x��״��Ϊ: [batch_size, vocabulary_size]
        log_ps = self.log_softmax(x)  # ����������ʣ���״: [batch_size, vocabulary_size]
        return log_ps

# Step 5: ��ʼѵ��
class config():
    """
    ѵ��������
    ����ģ�ͺ�ѵ���ĸ��ֲ���
    """
    def __init__(self):
        # ѵ������
        self.num_steps = 1000
        # ���δ�С
        self.batch_size = 128
        # ��������ÿ���ٲ���ӡһ����ʧ��
        self.check_step = 20
        
        # �ʻ���С
        self.vocabulary_size = 40000
        # ������ά��
        self.embedding_size = 200  # Ƕ��������ά��
        # �����Ĵ��ڴ�С
        self.skip_window = 4  # �������Ҹ����ٸ���
        # ÿ����������ɵı�ǩ����
        self.num_skips = 8  # �����������ɱ�ǩ�Ĵ���
        
        # �Ƿ�ʹ��GPU
        self.use_cuda = torch.cuda.is_available()
        
        # ѧϰ��
        self.lr = 0.03

# �������ö���
args = config()

# ����Skip-Gramģ��ʵ��
model = SkipGram(args)
# ��ӡģ�ͽṹ
print(model)
"""
Ԥ�����ʾ��:
SkipGram(
  (embedding): Embedding(40000, 128)
  (output): Linear(in_features=128, out_features=40000, bias=True)
  (log_softmax): LogSoftmax(dim=1)
)
"""

# ������ã���ģ���ƶ���GPU
if args.use_cuda:
    model = model.to('cuda')

# ���帺������Ȼ��ʧ�����������ڶ����������룩
nll_loss = nn.NLLLoss()
# ����Adam�Ż���
adam_optimizer = optim.Adam(model.parameters(), lr=args.lr)

# ��ӡ�ָ��ߺͿ�ʼѵ����Ϣ
print('-'*50)
print('Start training.')

# ��ʼ��ƽ����ʧ�Ϳ�ʼʱ��
average_loss = 0
start_time = time.time()

# ѵ��ѭ��
for step in range(1, args.num_steps):
    # ����ѵ������
    batch_inputs, batch_labels = generate_batch(
        data, args.batch_size, args.num_skips, args.skip_window)
    
    # ȥ����ǩ�ĵ�һά�ȣ���[batch_size, 1]��Ϊ[batch_size]��
    batch_labels = batch_labels.squeeze()
    
    # ��numpy����ת��ΪPyTorch����
    batch_inputs, batch_labels = torch.LongTensor(batch_inputs), torch.LongTensor(batch_labels)
    
    # ������ã��������ƶ���GPU
    if args.use_cuda:
        batch_inputs, batch_labels = batch_inputs.to('cuda'), batch_labels.to('cuda')
    
    # ǰ�򴫲�������ģ��������������ʣ�
    log_ps = model(batch_inputs)
    # ������ʧ
    loss = nll_loss(log_ps, batch_labels)
    # �ۼ���ʧ���ڼ���ƽ��ֵ
    average_loss += loss
    
    # �ݶ�����
    adam_optimizer.zero_grad()
    # ���򴫲��������ݶ�
    loss.backward()
    # �Ż�������������ģ�Ͳ���
    adam_optimizer.step()
    
    # ÿ��check_step����ӡһ��ѵ����Ϣ
    if step % args.check_step == 0:
        end_time = time.time()  # ��¼����ʱ��
        average_loss /= args.check_step  # ����ƽ����ʧ
        # ��ӡ��ǰ������ƽ����ʧ�ͺ�ʱ
        print('Average loss as step {0}: {1:.2f}, cost: {2:.2f}s.'.format(
            step, average_loss, end_time-start_time))
        # ���ÿ�ʼʱ���ƽ����ʧ
        start_time = time.time()
        average_loss = 0

"""
Ԥ�����ʾ��:
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

# ѵ�������ʾ
print('Training Done.')
print('-'*50)

# ��ȡѵ����Ĵ�Ƕ��Ȩ��
final_embedding = model.embedding.weight.data
# ��ӡǶ��������״
print(final_embedding.shape)
"""
Ԥ�����ʾ��:
torch.Size([40000, 200])
"""

# Step 6: ���ӻ���Ƕ��
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """
    ʹ��t-SNE��ά��Ĵ�Ƕ����ӻ�����
    ����:
        low_dim_embs: ��ά���Ƕ������
        labels: ��Ӧ�ĵ��ʱ�ǩ
        filename: ����ͼ����ļ���
    """
    # ���Ƕ�����������Ƿ��㹻
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    print('Visualizing.')
    
    # ������ߴ�ͼ��
    plt.figure(figsize=(18, 18))  # ��λ��Ӣ��
    
    # Ϊÿ���ʻ��Ƶ�ͱ�ǩ
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]  # ��ȡ�������
        plt.scatter(x, y)  # ����ɢ��
        
        # ����ı���ע
        plt.annotate(label,
                     xy=(x, y),  # ��ע��λ��
                     xytext=(5, 2),  # �ı�ƫ����
                     textcoords='offset points',  # �ı���������
                     ha='right',  # ˮƽ���뷽ʽ
                     va='bottom')  # ��ֱ���뷽ʽ
    
    # ����ͼ��
    plt.savefig(filename)
    print('TSNE visualization is completed, saved in {0}.'.format(filename))

# ����matplotlibʹ�÷ǽ���ʽ��ˣ�����ͼ����ʾ��
matplotlib.use("Agg")

# ����t-SNE��ά��
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# ����ֻ���ӻ�ǰ500����
plot_only = 500
# �Դ�Ƕ����н�ά
low_dim_embs = tsne.fit_transform(final_embedding[:plot_only, :])
# ��ȡ��Ӧ�ĵ��ʱ�ǩ
labels = [ idx2word[i ] for i in range(plot_only) ]
# ���ɿ��ӻ�ͼ��
plot_with_labels(low_dim_embs, labels, filename='./data/tsne.png')