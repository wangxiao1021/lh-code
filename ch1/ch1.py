#-*- coding:utf-8 -*-
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#在指定的间隔内返回均匀间隔的数字。返回num均匀分布的样本，在[start, stop]。

# 10个点
x = np.linspace(0, 1 ,10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪声的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

def fitting(M=0):
    '''
    :param M: 多项式的次数
    :return:
    '''

    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y)) # 三个参数：误差函数、函数参数列表、数据点
    print('Fitting Parametrs: ', p_lsq[0])

    #可视化
    plt.plot(x_points, real_func(x_points), label = 'real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label = 'noise')
    plt.legend()
    plt.show()
    return p_lsq

# M=0
p_lsq_0 = fitting(M=0)
# M=1
p_lsq_1 = fitting(M=1)
# M=3
p_lsq_3 = fitting(M=3)
# M=9
p_lsq_9 = fitting(M=9)


# M = 9 的时候出现过拟合，引入正则化项，降低过拟合
# 回归问题中，损失函数是平方损失，正则化可以是参数向量的L2范数，也可以是L1范数。
# L1 : regularization*abs(p)
# L2 : 0.5 * regularization * np.square(p)

regularization = 0.0001

def residuals_func_regularization(p, x, y):
    ret = fit_func(p,x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))  # L2 范数作为正则化项
    return ret

# 最小二乘法 ， 加正则化项
p_init = np.random.rand(9+1)
p_lsq_regularization = leastsq(residuals_func_regularization, p_init, args=(x, y))

plt.plot(x_points, real_func(x_points),label = 'real' )
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label ='fitted curve')
plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label = 'regularization')
plt.plot(x, y, 'bo', label = 'noise')
plt.legend()
plt.show()