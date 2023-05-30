# coding=utf-8

import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

from LinearRegression.linear_regression import LinearRegression
"""
    一个特征函数的线性回归
"""


# 读取数据
data = pd.read_csv('data/world-happiness-report-2017.csv')

# 得到训练和测试数据集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

# 显示原始数据集
# plt.scatter(x_train, y_train, label='Train data')
# plt.scatter(x_test, y_test, label='Test data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('Happy')
# plt.legend()
# plt.show()



# 训练模型
num_iterations = 500
learning_rate = 0.01
# 初始化数据
linea_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linea_regression.train(learning_rate, num_iterations)

# print(cost_history[0])
# print(cost_history[-1])


# 显示损失函数
# plt.plot(range(num_iterations), cost_history)
# plt.xlabel('Iter')
# plt.ylabel('cost')
# plt.title('GD')
# plt.show()


predictions_num = 100
# 随机取数据
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
# 计算预测回归结果
y_predictions = linea_regression.predict(x_predictions)

# 显示
plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='Test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()




















































