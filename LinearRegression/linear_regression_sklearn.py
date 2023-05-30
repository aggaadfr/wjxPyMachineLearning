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
from sklearn.linear_model import LinearRegression

"""
    sklearn实现线性回归
"""


# 1、手动实现线性回归方法
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
# 打印结果
print(theta_best)


# 构造新的数据
X_new = np.array([[0],[2]])
# 添加为1的一列
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)


plt.plot(X_new, y_predict, 'r--')
plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])
plt.show()




# 2、使用sklearn api
lin_reg = LinearRegression()
lin_val = lin_reg.fit(X, y)
print(lin_val.coef_)
print(lin_val.singular_)



# 3、批量梯度下降
eta = 0.1
n_iterations = 1000
m = 100
theta = np.random.rand(2, 1)
for iterations in range(n_iterations):
    # 计算梯度
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    # 更新学习率
    theta = theta - eta * gradients

print(theta)


# 最终的预测结果值
X_new_b.dot(theta)


# 4、测试学习率值大小的结果
theta_path_bgd = []
def plot_gradient_descent(theta, eta, theta_path = None):
    m = len(X_b)
    plt.plot(X, y, 'b.')
    n_iterations = 1000
    for iterations in range(n_iterations):
        y_predict = X_new_b.dot(theta)
        plt.plot(X_new, y_predict, "b-")
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('X_1')
    plt.axis([0,2,0,15])
    plt.title('eta = {}'.format(eta))


theta = np.random.rand(2, 1)
plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
plt.show()



# 5、随机梯度下降
theta_path_bgd=[]
m = len(X_b)
n_epochs = 50
t0 = 5
t1 = 50

# 衰减
def learning_schedule(t):
    return t0/(t1+t)

theta = np.random.rand(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch < 3 and i<10:
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, "r-")
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(n_epochs*m + i)
        theta = theta - eta * gradients
        theta_path_bgd.append(theta)

plt.plot(X, y, 'b.')
plt.axis([0,2,0,15])
plt.show()




# 小批量梯度下降
theta_path_mgd=[]
n_epochs = 50
minibatch = 16
theta = np.random.rand(2, 1)
# 指定随机种子
np.random.seed(0)

t = 0

for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch):
        t += 1
        xi = X_b_shuffled[i:i+minibatch]
        yi = y_shuffled[i:i+minibatch]
        gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
















