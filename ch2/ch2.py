#-*- coding:utf-8 -*-
# 感知机 二分类线性分类模型  f(x) = (w x + b )
# 分离超平面  w x + b = 0
# 最小化损失函数  min w,b  L(w,b) = - Σxi ∈ M  yi( w xi + b )
# 损失函数对应于误分类点到分离超平面的总距离


'''
感知机学习算法是基于随机梯度下降法的对损失函数的最优化算法，有原始形式和对偶形式。算法简单且易于实现。原始形式中，首先任意选取一个超平面，然后用梯度下降法不断极小化目标函数。在这个过程中一次随机选取一个误分类点使其梯度下降。
'''

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()
# 2    50
# 1    50
# 0    50

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')    # 0-50个
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')  #50-100
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])  #选前一百行的第一列 第二列 倒数第一列
X, y = data[:,:-1], data[:,-1]  # x的数据包括target之前的每一列， y是target那一列

# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 1.]
y = np.array([1 if i == 1 else -1 for i in y])

# [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
#  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
#  -1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1]

# Perceptron
# 数据线性可分，二分类数据
# 此处为一元一次线性方程

class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) -1, dtype=np.float32)  #长度去除target那一列  所以减1
        self.b = 0
        self.l_rate = 0.1
        # self.data = data
        print(self.w)

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    # 随机梯度下降
    def fit(self, X_train, y_train):
        is_Find = False
        while not is_Find:
            for d in range(len(X_train)):  # 对每一个训练数据
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:  #  分类错的样本  这里选的是第一个
                    self.w = self.w + self.l_rate * np.dot(y,X)
                    self.b = self.b + self.l_rate * y
                    break
                elif d == len(X_train) - 1 :
                    is_Find = True
            print('perceptron - one step')
        return is_Find

    def score(self):
        pass

perceptron = Model()
while(True):
    if(perceptron.fit(X, y)):
        break



print(perceptron.w[0])
print(perceptron.w[1])

x_points = np.linspace(4,7,10)
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
plt.plot(x_points, y_)


plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
clf.fit(X,y)

# Weights assigned to the features.
print(clf.coef_)

# 截距 Constants in decision function.
print(clf.intercept_)

x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
