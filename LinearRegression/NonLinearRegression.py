# coding=utf-8

import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

from LinearRegression.linear_regression import LinearRegression
"""
    非线性回归函数
"""


# 读取数据
data = pd.read_csv('data/non-linear-regression-x-y.csv')

x = data['x'].values.reshape((data.shape[0], 1))
y = data['y'].values.reshape((data.shape[0], 1))

data.head(10)


plt.plot(x, y)
plt.show()


# 训练模型
num_iterations = 50000
learning_rate = 0.02
# 特征复杂度
polynomial_degree = 15
# 非线性变化
sinusoid_degree = 15
normalize_data = True

# 初始化数据
linea_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)
(theta, cost_history) = linea_regression.train(learning_rate, num_iterations)

print(cost_history[0])
print(cost_history[-1])


theta_table = pd.DataFrame({'Model Parameters': theta.flatten()})

# 显示损失函数
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()



predictions_num = 1000
# 随机取数据
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)
# 计算预测回归结果
y_predictions = linea_regression.predict(x_predictions)

# 显示
plt.scatter(x, y, label='Train Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.show()




















































