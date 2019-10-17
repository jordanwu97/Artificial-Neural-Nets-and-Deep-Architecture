from _3_1 import *

from sklearn.metrics.pairwise import rbf_kernel

Y = np.arange(20)

X = np.arange(10)

var = 3

n = RBF_NET(Y, 3, sign)

phi = n.phi(X)


phi2 = rbf_kernel(X.reshape(-1,1), Y.reshape(-1,1), gamma=1/(2*var**2))

print (np.allclose(phi2,phi))