import numpy as np
from matplotlib import image

f = open("../orl_faces/s2/1.pgm", 'rb')
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
print(width, height, depth)
print(data)
image.imsave("../实验结果/picture4.png", data)


A = np.load("../实验结果/A_final.npy")
D = np.load("../实验结果/D_final.npy")
Y = np.dot(D, A)
y = Y[:, 0]
y = y.reshape(112, 92)
image.imsave("../实验结果/picture1_final.png", y)


