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
df = pd.DataFrame(iris.data, colomns = iris.feature_names)
df['label'] = iris.target

df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100], [0, 1, -1])