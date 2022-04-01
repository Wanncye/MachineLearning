# coding:utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image

def pca(data, n_dim):
    '''

    pca is O(D^3)
    :param data: (n_samples, n_features(D))
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    data = data - np.mean(data, axis = 0, keepdims = True)

    cov = np.dot(data.T, data)

    eig_values, eig_vector = np.linalg.eig(cov)
    # print(eig_values)
    indexs_ = np.argsort(-eig_values)[:n_dim]
    picked_eig_values = eig_values[indexs_]
    picked_eig_vector = eig_vector[:, indexs_]
    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim


# data 降维的矩阵(n_samples, n_features)
# n_dim 目标维度
# fit n_features >> n_samples, reduce cal
def highdim_pca(data, n_dim):
    '''

    when n_features(D) >> n_samples(N), highdim_pca is O(N^3)

    :param data: (n_samples, n_features)
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    N = data.shape[0]
    data = data - np.mean(data, axis = 0, keepdims = True)

    Ncov = np.dot(data, data.T)

    Neig_values, Neig_vector = np.linalg.eig(Ncov)
    indexs_ = np.argsort(-Neig_values)[:n_dim]
    Npicked_eig_values = Neig_values[indexs_]
    # print(Npicked_eig_values)
    Npicked_eig_vector = Neig_vector[:, indexs_]
    # print(Npicked_eig_vector.shape)

    picked_eig_vector = np.dot(data.T, Npicked_eig_vector)
    picked_eig_vector = picked_eig_vector/(N*Npicked_eig_values.reshape(-1, n_dim))**0.5
    # print(picked_eig_vector.shape)

    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim


def pca_img(data, k):
    # 求图片每一行的均值
    mean = np.array([np.mean(data[:, index]) for index in range(data.shape[1])])
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵：1/n＊(X * X^T)，这里不除以n也不影响
    matrix = np.dot(np.transpose(data), data)
    # 此函数可用来计算特征值及对应的特征向量
    # eig_val存储特征值，eig_vec存储对应的特征向量
    eig_val, eig_vec = np.linalg.eig(matrix)
    # 对矩阵操作，按从小到大的顺序对应获得此数的次序（从0开始）
    # 比如说有矩阵[2,1,3,－1]
    # 那么将按数组的顺序[－1,1，2，3］输出对应的下标
    # 即[2，1，3，0]
    eig_index = np.argsort(eig_val)
    # 取下标的倒数k位，也就是取前k个大特征值的下标
    eig_vec_index = eig_index[:-(k+1):-1]
    # 取前k个大特征值的特征向量
    feature = eig_vec[:, eig_vec_index]
    # 将特征值与对应特征向量矩阵乘得到最后的pca降维图
    new_data = np.dot(normal_data, feature)
    # 将降维后的数据映射回原空间
    rec_data = np.dot(new_data, np.transpose(feature)) + mean
    # 压缩后的数据也需要乘100还原成RGB值的范围
    newImage = Image.fromarray(np.uint8(rec_data*100))
    # 将处理好的降维图片存入文件夹
    newImage.convert('RGB').save('k=' + str(k) + '.jpg')
    newImage.show()

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    Y = data.target
    data_2d1 = pca(X, 2)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.title("my_PCA")
    plt.scatter(data_2d1[:, 0], data_2d1[:, 1], c = Y)

    sklearn_pca = PCA(n_components=2)
    data_2d2 = sklearn_pca.fit_transform(X)
    plt.subplot(122)
    plt.title("sklearn_PCA")
    plt.scatter(data_2d2[:, 0], data_2d2[:, 1], c = Y)
    plt.savefig("PCA.png")
    plt.show()


    png_name = '809'
    img = Image.open(png_name + '.png')
    img = np.copy(img)
    img_origin_shape = img.shape
    print(img.shape)
    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.imshow(img)
    plt.title("origin")

    img = img.reshape(-1,3)
    sklearn_pca_img = PCA(n_components=2)
    img = sklearn_pca_img.fit_transform(img)
    img = sklearn_pca_img.inverse_transform(img)
    img = img.reshape(img_origin_shape).astype(np.uint8)
    # img_r = sklearn_pca_img.fit_transform(img[:,:,0])
    # img_r = sklearn_pca_img.inverse_transform(img_r)
    # img_g = sklearn_pca_img.fit_transform(img[:,:,1])
    # img_g = sklearn_pca_img.inverse_transform(img_g)
    # img_b = sklearn_pca_img.fit_transform(img[:,:,2])
    # img_b = sklearn_pca_img.inverse_transform(img_b)
    # img = np.stack((img_r,img_g,img_b), axis=2).astype(np.uint8)
    print(img.shape)
    plt.subplot(122)
    plt.title("after pca")
    plt.imshow(img)
    plt.savefig(png_name + "_pca.png")
