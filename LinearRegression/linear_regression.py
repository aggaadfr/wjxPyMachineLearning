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
    def __int__(self, data, labels, polynomial_degree = 0, sinusoid_degree = 0, normalize_data=True):
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



