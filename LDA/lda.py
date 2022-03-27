#coding:utf-8
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
'''
输入：数据集𝐷={(𝑥1,𝑦1),(𝑥2,𝑦2),...,((𝑥𝑚,𝑦𝑚))},其中任意样本𝑥𝑖为n维向量，𝑦𝑖∈{𝐶1,𝐶2,...,𝐶𝑘}，降维到的维度d。

　　　　输出：降维后的样本集D′

　　　　1) 计算类内散度矩阵𝑆𝑤

　　　　2) 计算类间散度矩阵𝑆𝑏

　　　　3) 计算矩阵𝑆^(−1)𝑤𝑆𝑏

　　　　4）计算𝑆−1𝑤𝑆𝑏的最大的d个特征值和对应的d个特征向量(𝑤1,𝑤2,...𝑤𝑑),得到投影矩阵W𝑊

　　　　5) 对样本集中的每一个样本特征𝑥𝑖,转化为新的样本𝑧𝑖=𝑊𝑇𝑥𝑖

　　　　6) 得到输出样本集𝐷′={(𝑧1,𝑦1),(𝑧2,𝑦2),...,((𝑧𝑚,𝑦𝑚))}

'''

def lda(data, target, n_dim):
    '''
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''

    clusters = np.unique(target)

    if n_dim > len(clusters)-1:
        print("K is too much")
        print("please input again")
        exit(0)

    #within_class scatter matrix 类内散度矩阵
    Sw = np.zeros((data.shape[1],data.shape[1]))
    for i in clusters: 
        datai = data[target == i]
        datai = datai-datai.mean(0)
        Swi = np.mat(datai).T*np.mat(datai)
        Sw += Swi

    #between_class scatter matrix 类间散度矩阵
    SB = np.zeros((data.shape[1],data.shape[1]))
    u = data.mean(0)  #所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  #某个类别的平均值
        SBi = Ni*np.mat(ui - u).T*np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw)*SB #np.linalg.inv()：矩阵求逆  np.linalg.det()：矩阵求行列式（标量）
    eigVals,eigVects = np.linalg.eig(S)  #求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim-1):-1]
    w = eigVects[:,eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    data_1 = lda(X, Y, 2)

    data_2 = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, Y)


    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_LDA")
    plt.scatter(data_1[:, 0], data_1[:, 1], c = Y)

    plt.subplot(122)
    plt.title("sklearn_LDA")
    plt.scatter(data_2[:, 0], data_2[:, 1], c = Y)
    plt.savefig("LDA.png")
    plt.show()