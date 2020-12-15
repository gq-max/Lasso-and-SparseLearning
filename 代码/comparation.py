from load_data import *
import numpy as np


def loss_function(X, D, A):
    one = X - np.dot(D, A)
    loss_mat = np.dot(one.T, one)
    loss = sum(sum(loss_mat))
    return loss


X = get_data()

init_D = get_init_D()

p, n = X.shape
p, k = init_D.shape
init_A = np.random.randn(k, n)

A1 = np.load("../实验结果/A.npy")
D1 = np.load("../实验结果/D.npy")

A2 = np.load("../实验结果/A_new.npy")
D2 = np.load("../实验结果/D_new.npy")

A3 = np.load("../实验结果/A_final.npy")
D3 = np.load("../实验结果/D_final.npy")

init_loss = loss_function(X, init_D, init_A)
loss1 = loss_function(X, D1, A1)
loss2 = loss_function(X, D2, A2)
loss3 = loss_function(X, D3, A3)

print("最初的损失函数为：", init_loss)
print("循环1*1次的损失函数为：", loss1)
print("循环2*3次的损失函数为：", loss2)
print("循环2*5次的损失函数为：", loss3)
