import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import torch
from torch import nn 
from torch.utils import data
from torch.nn import functional as F 

import torchvision
from torchvision import transforms

import collections
import re
from d2l import torch as d2l
import random

def plot(X, Y, labels = None, xlabel = None, ylabel = None, title = None, 
         grid = True, scatter = False, save = False):
    '''
    use to draw curves\\
    Y: tuple of array\\
    labels: tuple of label\\
    '''
    plt.clf()
    if scatter:
        draw_fn = plt.scatter
    else:
        draw_fn = plt.plot
        
    if type(Y) == tuple or type(Y) == list:
        for i in range(len(Y)):
            if labels is not None:
                draw_fn(X, Y[i], label = labels[i])
            else:
                draw_fn(X, Y[i])
    else:
        if labels is not None:
            draw_fn(X, Y, label = labels)
        else:
            draw_fn(X, Y)
    if labels is not None:
        plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid()
    if save:
        plt.savefig('./' + title + '.png')

def load_array(arrays, batch_size):
    dataset = data.TensorDataset(*arrays)
    return data.DataLoader(dataset, batch_size)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    '''draw picture in fashion mnist'''
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def load_fashion_mnist(batch_size, resize = None):
    '''Download Fashion MNIST, and load it to memory'''
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = './data',
        train = True,
        transform = trans,
        download = True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = './data',
        train = False,
        transform = trans,
        download = True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))

def accuracy(y_hat, y):
    '''calculate the number of right predictions'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.sum()

def evaluate_accuracy(model, data_iter):
    if isinstance(model, nn.Module):
        model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(model(X), y), y.numel())    
    return metric[0] / metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def generate_data(w_true, b_true, num):
    '''use true value get datum with noise'''
    X = torch.normal(0, 1, (num, len(w_true)))
    Y = X @ w_true.reshape(-1, 1) + b_true
    noise = torch.normal(0, 0.001, Y.shape)
    Y += noise
    return X, Y.reshape(-1)

def LinearReg(X, w, b):
    return X @ w.reshape(-1, 1) + b

def SGD(params, lr):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()
            
def MSE(y_hat, y):
    return (0.5 * (y_hat - y.reshape(y_hat.shape))**2).mean()

def CrossEntropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y]).mean()

def softmax(o):
    return torch.exp(o)/torch.exp(o).sum(dim = 1).reshape(-1, 1)

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine(): 
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'): 
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

class Timer: 
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EpochRecorder:
    def __init__(self, n):
        self.n = n
        self.memory = []
        for _ in range(self.n):
            self.memory.append([])
           
    def append(self, *args):
        for i, num in enumerate(args):
            self.memory[i].append(num)
    
    def reset(self):
        self.memory = []
        for _ in range(self.n):
            self.memory.append([])
            
    def plot(self, labels = None, xlabel = 'epochs', ylabel = 'metric', title = None, 
            grid = True, scatter = False, save = False):
        epochs = list(range(len(self.memory[0])))
        plot(epochs, self.memory, labels, xlabel, ylabel, title, grid, scatter, save)
        
    def get_num(self, idx, mode = 'mean'):
        if mode == 'mean':
            return float(np.mean(self.memory[idx]))
        elif mode == 'sum':
            return float(np.sum(self.memory[idx]))
        else:
            return 'fuck'
            
    def __getitem__(self, idx):
        return self.memory[idx]

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader: 
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

if __name__ == '__main__':
    rec = EpochRecorder(3)
    rec.append(3, 4, 5)
    rec.append(4, 5, 6)
    rec.append(5, 6, 7)
    rec.append(7, 8, 9)
    rec.plot(labels = ['1', '2', '3', '4'], title='test')
    plt.show()    