import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

# a)
root = '/home/tuomas/Python/DATA.ML.200/Ex1/'
Z = img.imread(root+'uneven_illumination.jpg')
plt.imshow(Z, cmap='gray')

Z_ = Z

X, Y = np.meshgrid(range(1300), range(1030))

# b)
X = X.ravel()
Y = Y.ravel()
Z = Z.ravel()

# c)
H = np.array([X*X, Y*Y, X*Y, X, Y, np.ones(X.shape[0])]).transpose()

c = np.linalg.inv(np.matmul(H.transpose(), H))
c = np.matmul(c, H.transpose())
c = np.matmul(c, Z)

# d)
Z_pred = np.dot(H, c)
Z_pred = Z_pred.reshape((1030, 1300))
Z = Z.reshape((1030, 1300))

# e)
Z_res = Z - Z_pred
plt.imshow(Z_res, cmap='gray')
