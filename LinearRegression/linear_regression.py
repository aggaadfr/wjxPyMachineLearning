# coding=utf-8
import numpy as np
from utils.features import prepare_for_training

# 构建线性回归
class LinearRegression:
    """
        1、对数据进行预处理
        2、先得到所有的特征个数
        3、初始化参数矩阵
    """
    def __init__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data=True):
        """
            数据初始化操作
        :param data: 训练数据集
        :param labels: 训练的结果数据集
        :param polynomial_degree:
        :param sinusoid_degree:
        :param normalize_data:
        :return:
        """
        # 数据预处理
        (data_processed, features_mean, features_deviation) = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations = 500):
        """
            训练模块，执行梯度下降
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return:
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history



    def gradient_descent(self, alpha, num_iterations):
        cost_history = []
        """
            梯度下降
        :param alpha:
        :param num_iterations:
        :return:
        """
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history


    def gradient_step(self, alpha):
        """
            梯度下降参数更新方法，矩阵运算
        :param alpha:
        :return:
        """
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothsis(self.data, self.theta)
        delta = prediction - self.labels
        theta = self.theta
        # 算法公式
        theta = theta- alpha*(1/num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta


    def cost_function(self, data, labels):
        """
            损失计算方法
        :param data:
        :param labels:
        :return:
        """
        num_examples = data.shape[0]
        delta = LinearRegression.hypothsis(self.data, self.theta) - labels
        # 损失函数
        cost = (1/2) * np.dot(delta.T, delta)
        # print(cost.shape)
        return cost[0][0]

    @staticmethod
    def hypothsis(data, theta):
        """
            预测函数
        :param data: 当前预测值
        :param theta:
        :return:
        """
        # 矩阵乘法
        return np.dot(data, theta)




    def get_cost(self, data, labels):
        """
            得到当前损失值
        :param data:
        :param labels:
        :return:
        """
        data_processed = prepare_for_training(data, self.polynomial_degree,self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_processed, labels)


    def predict(self, data):
        """
            用训练好的参数模型，用预测得到回归结果
        :param data:
        :return:
        """
        data_processed = prepare_for_training(data, self.polynomial_degree,self.sinusoid_degree, self.normalize_data)[0]
        return LinearRegression.hypothsis(data_processed, self.theta)

















