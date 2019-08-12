#-*-coding:utf-8-*-
# ������ѧϰ�㷨����3���֣�����ѡ���������ɺ����ļ�֦�����õ��㷨��ID3�� C4.5��CART��
#coding=utf8
import sys
reload(sys)
sys.setdefaultxxxx("utf8")
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
    print(ent)
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
    return cond_ent()


# ��Ϣ����
def info_gain(ent, cond_ent):
    return  ent-cond_ent

def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent,cond_ent(datasets,axis=c))
        best_feature.append((c,c_info_gain))
        print('����({}) - info_gain - {:.3f}').format(labels[c], c_info_gain)
    #�Ƚϴ�С
        best_ = max(best_feature, key=lambda x:x[-1])
        return '����({})����Ϣ�������ѡ��Ϊ���ڵ�����'.format(labels[best_[0]])

info_gain_train(np.array(datasets))

