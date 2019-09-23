import numpy as np

x1=[-1, -1, 1, -1, 1, -1, -1, 1]
x2=[-1, -1, -1, -1, -1, 1, -1, -1]
x3=[-1, 1, 1, -1, -1, 1, -1, 1]


X = np.vstack((x1,x2,x3))


M = X.shape[1]

W = np.zeros((M,M))

for x in X:
  W += np.outer(x,x)

np.fill_diagonal(W,0)

for x in X:
  print (x)
  xj = np.sign(np.matmul(W,x))
  print (np.all(x==xj))