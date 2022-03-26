# --coding:utf-8--
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
mu = 30  # mean of distribution
sigma = 2  # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)


def mle(x):
    """
    极大似然估计
    如果假定随机变量服从某种分布（如正态分布），可以通过统计手段来计算该分布的参数
    极大似然估计（Maximum Likelihood Estimate, MLE）是一种参数估计的方法，利用已知样本结果，反推最大可能导致这样结果的参数值。
    :param x:
    :return:
    """
    u = np.mean(x)
    return u, np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])


print(mle(x))
num_bins = 100
plt.hist(x, num_bins)
plt.show()