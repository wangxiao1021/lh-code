#-*-coding:GB18030-*-
# ������ѧϰ�㷨����3���֣�����ѡ���������ɺ����ļ�֦�����õ��㷨��ID3�� C4.5��CART��

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
import pprint


def create_data():
    datasets = [['����', '��', '��', 'һ��', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', 'һ��', '��'],
                ['����', '��', '��', 'һ��', '��'],
                ['����', '��', '��', 'һ��', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', '�ǳ���', '��'],
                ['����', '��', '��', '�ǳ���', '��'],
                ['����', '��', '��', '�ǳ���', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', '��', '��'],
                ['����', '��', '��', '�ǳ���', '��'],
                ['����', '��', '��', 'һ��', '��'],
                ]
    labels = [u'����', u'�й���', u'���Լ��ķ���', u'�Ŵ����', u'���']
    # �������ݼ���ÿ��ά�ȵ�����
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)

#shang
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum( [ (p/data_length) * log(p/data_length,2)    for p in label_count.values()])
    return ent

# ����������
def cond_ent(datasets, axis = 0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length) * calc_ent(p) for p in feature_sets.values()])
    return cond_ent


# ��Ϣ����
def info_gain(ent, cond_ent):
    return  ent-cond_ent

def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
#     ent = entropy(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('����({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))
    # �Ƚϴ�С
    best_ = max(best_feature, key=lambda x: x[-1])
    return '����({})����Ϣ�������ѡ��Ϊ���ڵ�����'.format(labels[best_[0]])

print(info_gain_train(np.array(datasets)))

# ID3
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature = None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label': self.label,
            'feature':self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features) #???

class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # shang
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = data_length[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] +=1
        ent = -sum([(p/data_length) * log(p/data_length, 2) for p in label_count.values()])
        return ent

    # ����������
    def cond_ent(self, datasets, axis =0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(p / data_length)* self.calc_ent(p)] for p in feature_sets.values())
        return cond_ent

    # ��Ϣ����
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        feature_count = len(datasets[0] - 1)
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(feature_count):
            c_info_gain  = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # �Ƚϴ�С
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        '''
        :param train_data:  ���ݼ�D(DataFrame��ʽ), ������A, ��ֵeta
        :return: ������T
        '''

        _, y_train, features = train_data.iloc[:, :-1], \
                               train_data.iloc[:, -1], \
                               train_data.columns[: -1]

        # 1,��D��ʵ������ͬһ��Ck����TΪ���ڵ�����������Ck��Ϊ�������ǣ�����T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2, ��AΪ�գ���TΪ���ڵ�������D��ʵ����������Ck��Ϊ�ýڵ�����ǣ�����T
        if len(features) == 0:
            return Node(
                root=True,
                label = y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,���������Ϣ���� ͬ5.1,AgΪ��Ϣ������������
        max_feature, max_infro_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag����Ϣ����С����ֵeta,����TΪ���ڵ���������D����ʵ����������Ck��Ϊ�ýڵ�����ǣ�����T
        if max_infro_gain < self.epsilon:
            return Node(
                root=True,
                label = y_train.value_counts().sort_values(ascending=False).index[0]
            )

        # 5,����Ag�Ӽ�
        node_tree = Node(
            root=False, feature_name=max_feature_name, feature=max_feature
        )

