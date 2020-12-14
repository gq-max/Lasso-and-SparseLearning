import numpy as np
from sklearn import preprocessing


def read_pgm(filename):
    '''
    读取pgm文件内容，并将其转换为行向量
    :param filename: 文件名
    :return: 10304个特征的样本值
    '''
    f = open(filename, 'rb')
    f.readline()  # P5\n
    (width, height) = [int(i) for i in f.readline().split()]
    depth = int(f.readline())
    data = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(ord(f.read(1)))
        data.append(row)
    data = np.array(data)
    data = data.reshape(width * height)
    return data


def get_data():
    """得到数据集X，其中X的每列为一个样本， 每一行为一个特征，X为p*n的矩阵"""
    X = []
    for i in range(1, 41):
        for j in range(1, 11):
            fn = "../orl_faces/s{}/{}.pgm".format(i, j)
            data = read_pgm(fn)
            X.append(data)
    X = np.array(X)
    X = X.T
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    return X


def get_init_D():
    """得到初始的字典D, D为p*k维的矩阵"""
    D = []
    for i in range(1, 41):
        j = np.random.randint(1, 10)
        fn = "../orl_faces/s{}/{}.pgm".format(i, j)
        data = read_pgm(fn)
        D.append(data)
    D = np.array(D)
    D = D.T
    D_mean = np.mean(D, axis=0)
    D_std = np.std(D, axis=0)
    D = (D - D_mean) / D_std
    return D
