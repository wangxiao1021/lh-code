#-*-coding:GB18030-*-
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['lanel'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    return data[:,:2], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class LogisticReressionClassifier:
    def __init__(self, max_iter=200, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate


    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])  # ��������
        return data_mat

    def fit(self, X, y):
        # label = np.mat(y)
        data_mat = self.data_matrix(X)  # m*n

        self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)
        print(self.weights)
        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
        print('LogisticRegression Model(learning_rate={},max_iter={})'.format(
            self.learning_rate, self.max_iter))

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)



lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_test, y_test))

x_points = np.arange(4, 8)
y_ = -(lr_clf.weights[1]*x_points + lr_clf.weights[0]) / lr_clf.weights[2]
plt.plot(x_points, y_)  # ����ƽ��

#lr_clf.show_graph()
plt.scatter(X[:50,0],X[:50,1], label='0')
plt.scatter(X[50:,0],X[50:,1], label='1')
plt.legend()
plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.coef_, clf.intercept_)

x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(X[:50, 0], X[:50, 1], 'bo', color='blue', label='0')
plt.plot(X[50:, 0], X[50:, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

# �����
import math
from copy import deepcopy
class MaxEntropy:
    def __init__(self, EPS=0.005):
        self.samples = []
        self._Y = set()  # ��ǩ���ϣ��൱ȥȥ�غ��y
        self._numXY = {}  # keyΪ(x,y)��valueΪ���ִ���
        self._N = 0  # ������
        self._Ep_ = []  # �����ֲ�����������ֵ
        self._xyID = {}  # key��¼(x,y),value��¼id��
        self._n = 0  # ������ֵ(x,y)�ĸ���
        self._C = 0  # ���������
        self._IDxy = {}  # keyΪ(x,y)��valueΪ��Ӧ��id��
        self._w = []
        self._EPS = EPS  # ��������
        self._lastw = []  # ��һ��w����ֵ

    def loadData(self, dataset):
        self._samples = deepcopy(dataset)
        for items in self._samples:
            y = items[0]
            X = items[1:]
            self._Y.add(y)  # ���������Ѿ����������ǩ����Զ�����
            for x in X:
                if(x, y) in self._numXY:
                    self._numXY[(x, y)] += 1
                else:
                    self._numXY[(x, y)] = 1

        self._N = len(self._samples) #���ٸ�����
        self._n = len(self._numXY)  #����������ֵ
        self._C = max([len(sample) - 1 for sample in self._samples])
        self._w = [0] * self._n #???
        self._lastw = self._w[:]

        self._Ep_ = [0] * self._n
        # enumerate() �������ڽ�һ���ɱ��������ݶ���(���б�Ԫ����ַ���)���Ϊһ���������У�ͬʱ�г����ݺ������±꣬һ������ for ѭ�����С�
        for i, xy in enumerate(self._numXY): # ������������fi���ھ���ֲ�������
            self._Ep_[i] = self._numXY[xy] / self._N
            self._xyID[xy] = i
            self._IDxy[i] = xy

    def _Zx(self, X):  # ����ÿ��Z(x)��ֵ  �����õİ�
        zx = 0
        for y in self._Y:
            ss = 0
            for x in X:
                if(x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]  #???
            zx += math.exp(ss)
        return zx

    def _model_pyx(self, y, X):  # ����ÿ��P(y|x)
        zx = self._Zx(X)
        ss = 0
        for x in X:
            if (x, y) in self._numXY:
                ss += self._w[self._xyID[(x, y)]]   #???
        pyx = math.exp(ss) / zx
        return pyx

#  ����û��
    def _model_ep(self, index):  # ������������fi����ģ�͵�����
        x, y = self._IDxy[index]
        ep = 0
        for sample in self._samples:
            if x not in sample:
                continue
            pyx = self._model_pyx(y, sample)
            ep += pyx / self._N
        return ep

    def _convergence(self):  # �ж��Ƿ�ȫ������
        for last, now in zip(self._lastw, self._w):
            if abs(last - now) >= self._EPS:
                return False
        return True

    def predict(self, X):  # ����Ԥ�����
        Z = self._Zx(X)
        result = {}
        for y in self._Y:
            ss = 0
            for x in X:
                if (x, y) in self._numXY:
                    ss += self._w[self._xyID[(x, y)]]
            pyx = math.exp(ss) / Z
            result[y] = pyx
        return result

    def train(self, maxiter=1000):  # ѵ������
        for loop in range(maxiter):  # ���ѵ������
            print("iter:%d" % loop)
            self._lastw = self._w[:]
            for i in range(self._n):
                ep = self._model_ep(i)  # �����i��������ģ������
                self._w[i] += math.log(self._Ep_[i] / ep) / self._C  # ���²���
            print("w:", self._w)
            if self._convergence():  # �ж��Ƿ�����
                break





dataset = [['no', 'sunny', 'hot', 'high', 'FALSE'],
           ['no', 'sunny', 'hot', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'high', 'FALSE'],
           ['yes', 'rainy', 'mild', 'high', 'FALSE'],
           ['yes', 'rainy', 'cool', 'normal', 'FALSE'],
           ['no', 'rainy', 'cool', 'normal', 'TRUE'],
           ['yes', 'overcast', 'cool', 'normal', 'TRUE'],
           ['no', 'sunny', 'mild', 'high', 'FALSE'],
           ['yes', 'sunny', 'cool', 'normal', 'FALSE'],
           ['yes', 'rainy', 'mild', 'normal', 'FALSE'],
           ['yes', 'sunny', 'mild', 'normal', 'TRUE'],
           ['yes', 'overcast', 'mild', 'high', 'TRUE'],
           ['yes', 'overcast', 'hot', 'normal', 'FALSE'],
           ['no', 'rainy', 'mild', 'high', 'TRUE']]

maxent = MaxEntropy()
maxent.loadData(dataset)
maxent.train()
x = ['overcast', 'mild', 'high', 'FALSE']
print('predict:', maxent.predict(x))