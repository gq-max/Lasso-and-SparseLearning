from load_data import *

X = get_init_D()
Y = get_data()
y = Y[:, 0:1]
w = np.random.randn(40, 1)


def loss_function(y, X, w, lamda=1):
    one = y - np.dot(X, w)
    loss = np.dot(one.T, one)
    return loss


loss = loss_function(y, X, w)
print(loss)



# m, n = D.shape
# x = 0
# y = 0
# for i in range(n):
#     x += D[0, i]
# print(x)
# for j in range(m):
#     y += D[j, 0]
# print(y)
# print(np.mean(D[0, :]))
# print((np.mean(D[:, 0])))
# # d = D[:, 1]
# # two = sum([np.abs(i) for i in d])
# # one = np.linalg.norm(d, ord=1)
# # print(one, two)
