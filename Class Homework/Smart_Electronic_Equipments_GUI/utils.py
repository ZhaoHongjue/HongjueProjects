import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# 存储数据
class Data:
    def __init__(self):
        self.i = 0
        self.df = pd.DataFrame(columns=['time', '1_T','1_H','1_L',
                                        '2_T','2_H','2_L',
                                        '3_T','3_H','3_L',
                                        '4_T','4_H','4_L',])
        self.df.loc[0, 'time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.df.iloc[0, 1:] = np.ones(12) * -1
        self.bias = np.zeros(12)

    def store(self, data):
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.df.loc[self.i, 'time'] = time_now
        self.df.iloc[self.i, 1:] = data - self.bias
        self.i = self.i + 1

    def set_bias(self, bias):
        self.bias = bias

    def update(self):
        self.df.iloc[:, 1:] -= self.bias
    
    def get_new(self):
        return self.df.iloc[self.i-1].values
    
    def get_node(self, node):
        return self.df[['time', str(node) + '_T', str(node) + '_H', str(node) + '_L']]
    
    def get_param(self, param):
        return self.df[['time', '1_' + param, '2_' + param, '3_' + param, '4_' + param]]

    def clear(self, save = True):
        temp = self.get_new()
        if save:
            self.save()
        self.df = pd.DataFrame(columns=['time','1_T','1_H','1_L',
                                        '2_T','2_H','2_L',
                                        '3_T','3_H','3_L',
                                        '4_T','4_H','4_L',])
        self.i = 0
        self.df.loc[self.i] = temp
        self.i += 1
        self.save(name='running')

    def save(self, name = None):
        if name is None:
            name = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.df.to_csv('./data/' + name + '.csv')

    def load(self, name):
        self.df = pd.read_csv('./data/' + name + '.csv', index_col = 0)