
from load_data import *


def loss_function(y, X, w, lamda=1):
    one = y - np.dot(X, w)
    loss = np.dot(one.T, one)
    return loss


def coordinate_descent(y, X, w, lamda=1):
    n, p = X.shape
    loss = loss_function(y, X, w)
    # 使用坐标下降法优化回归系数alpha
    for it in range(10):
        for k in range(p):
            b_k = sum([(X[i, k] ** 2) for i in range(n)])
            a_k = 0
            for i in range(n):
                a_k += X[i, k] * (sum([(X[i, j] * w[j]) for j in range(p) if j != k]) - y[i])
            if a_k < -lamda / 2:
                w_k = -(a_k + lamda / 2) / b_k
            elif a_k > lamda / 2:
                w_k = -(a_k - lamda / 2) / b_k
            else:
                w_k = 0
            w[k] = w_k
        loss_prime = loss_function(y, X, w)
        print("loss:", loss_prime)
        delta = abs(loss_prime - loss)
        loss = loss_prime
        if delta < 0.1:
            break


def update_D(X, D, A, lamda=1):
    """更新字典D"""
    det_number = np.linalg.det(np.dot(A, A.T))
    if det_number:
        one = np.dot(X, A.T)
        two = np.linalg.inv(np.dot(A, A.T))
        D[:, :] = np.dot(one, two)
    else:
        m, n = np.dot(A, A.T).shape
        I = np.identity(m)
        one = np.dot(X, A.T)
        two = np.linalg.inv(np.dot(A, A.T) + lamda * I)
        D[:, :] = np.dot(one, two)


def sparse_learning(X, D, A, lamda=1, max_iter=10):
    """交替迭代求解D和A"""
    p, n = X.shape
    for iter in range(max_iter):
        print("第{}次迭代".format(iter))
        for i in range(n):
            print("优化A的第{}列".format(i))
            coordinate_descent(X[:, i], D, A[:, i], lamda=lamda)
        update_D(X, D, A, lamda=lamda)


def init():
    X = get_data()
    D = get_init_D()
    p, n = X.shape
    p, k = D.shape
    A = np.random.randn(k, n)
    return X, D, A


X, D, A = init()
sparse_learning(X, D, A, max_iter=2)
np.save('../实验结果/D_final.npy', D)
np.save('../实验结果/A_final.npy', A)
print(A, "\n", D)
