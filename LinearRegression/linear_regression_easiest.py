# coding=utf-8

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
import warnings
warnings.filterwarnings('ignore')

"""
    最简单实现线性回归
"""

# 创造模拟的抖动数据点
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

# plt.plot(X, y, 'b.')
# plt.xlabel('X_1')
# plt.ylabel('y')
# plt.axis([0,2,0,15])
# plt.show()

# 拼接一列1的数
X_b = np.c_[np.ones((100,1)), X]
# 对数据进行转至
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 构造新的数据
X_new = np.array([[0],[2]])
# 添加为1的一列
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)


plt.plot(X_new, y_predict, 'r--')
plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])
plt.show()




print(theta_best)









