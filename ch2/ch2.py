# ��֪�� ���������Է���ģ��  f(x) = (w x + b )
# ���볬ƽ��  w x + b = 0
# ��С����ʧ����  min w,b  L(w,b) = - ��xi �� M  yi( w xi + b )
# ��ʧ������Ӧ�������㵽���볬ƽ����ܾ���


'''
��֪��ѧϰ�㷨�ǻ�������ݶ��½����Ķ���ʧ���������Ż��㷨����ԭʼ��ʽ�Ͷ�ż��ʽ���㷨��������ʵ�֡�ԭʼ��ʽ�У���������ѡȡһ����ƽ�棬Ȼ�����ݶ��½������ϼ�С��Ŀ�꺯���������������һ�����ѡȡһ��������ʹ���ݶ��½���
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