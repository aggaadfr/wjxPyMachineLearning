# coding=utf-8

import numpy as np
import  pandas as pd
import plotly
import plotly.graph_objs as go
import webbrowser

from LinearRegression.linear_regression import LinearRegression

plotly.offline.init_notebook_mode()
"""
    多个特征函数的线性回归
"""


# 读取数据
data = pd.read_csv('data/world-happiness-report-2017.csv')

# 得到训练和测试数据集
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name_1, input_param_name_2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1, input_param_name_2]].values
y_test = test_data[[output_param_name]].values

plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line':{
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line':{
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name},
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)
# 使用plotly绘制3D散点图
# plot_data = [plot_training_trace, plot_test_trace]
# plot_figure = go.Figure(data=plot_data, layout=plot_layout)
# plotly.offline.plot(plot_figure)
# plotly.offline.iplot(plot_figure)
# 打开新页面
# webbrowser.open_new_tab('temp-plot.html')



# 训练模型
num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0
# 初始化数据
linea_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linea_regression.train(learning_rate, num_iterations)

print(cost_history[0])
print(cost_history[-1])



predictions_num = 100

x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()

y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()

# 计算平均值
x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)
# 初始化数组
x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

# 随机取数据
x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

# 计算预测回归结果
z_predictions = linea_regression.predict(np.hstack((x_predictions, y_predictions)))
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

# 绘制3D图
plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
webbrowser.open_new_tab('temp-plot.html')

















































