from scipy.io import loadmat
import matplotlib.pyplot as plt

root = '/home/tuomas/Python/DATA.ML.200/Ex1/'
mat = loadmat(root+'twoClassData.mat')

print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()

X0 = X[y==0,:]
X1 = X[y==1,:]
plt.plot(X0[:, 0], X0[:, 1], 'or')
plt.plot(X1[:, 0], X1[:, 1], 'ob')

